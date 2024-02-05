"""
Feb 03, 2024 @suncerock

This file is used to generate audio-segment pair metadata
which will then be used to train/evaluate the auto-segmentation system.

The main function is generate_multi_group().
You should NEVER call the other two functions explicitly!!!
"""

import os
from itertools import product
from tqdm import tqdm

import pandas as pd

from utils.default_configs_path import (
    audio_dir,
    segment_dir,
    segment_status_csv
)

def _write_audio_seg_csv(audio_seg, output_csv):
    f = open(output_csv, 'w')
    for stu_id, audio_file, segment_file in audio_seg:
        f.write("{},{},{}\n".format(stu_id, audio_file, segment_file))
    f.close()

def _generate(
    root_audio_dir,
    root_segment_dir,
    output_dir,
    year,
    band,
    instrument,
    summary_df
):
    """
    This is an internal function that should not be called by users.
    Generate audio-segment pair metadata for one (year, band, instrument) group.
    Write the results into a .csv file in the output directory.

    Parameters
    ----------
    root_audio_dir : str
        root directory of the audio files
    root_segment_dir : str
        root directory of the manually labeled segmentation files
    output_dir : str
        directory to output the metadata
    year : int
        the year of the group
    band : str
        the band of the group
    instrument : str
        the instrument of the group
    summary_df : pd.DataFrame
        the DataFrame of the group
        Note that the DataFrame has already been filtered by (year, band, instrument),
        and thus the (year, band, instrument) are used only for naming the output file and finding the folders
    
    Return
    ----------
    None

    """
    audio_folder = os.path.join(root_audio_dir, "bystudent", str(year), band)
    segment_folder = os.path.join(root_segment_dir, "bystudent", str(year), band)

    summary_df = summary_df[summary_df.has_instrument & summary_df.has_segment]
    
    if len(summary_df) == 0:
        return
    
    stu_id_list = sorted(summary_df.StudentID.unique())
    id_audio_seg = []

    for stu_id in stu_id_list:
        audio_file = os.path.join(audio_folder, "{}/{}.mp3".format(stu_id, stu_id))
        segment_file = os.path.join(segment_folder, "{}/{}_seginst.csv".format(stu_id, stu_id))

        if not os.path.exists(audio_file):
            print("Missing audio: ", audio_file)
            continue
        if not os.path.exists(segment_file):
            print("Missing segmentation: ", segment_file)
            continue
        id_audio_seg.append((stu_id, audio_file, segment_file))

    if not os.path.exists(output_dir):
        print("Missing output folder: {}! Making...".format(output_dir))
        os.mkdir(output_dir)

    output_csv = os.path.join(output_dir, "{}_{}_{}_audio_seg.csv".format(year, band, instrument.replace(" ", "")))
    _write_audio_seg_csv(id_audio_seg, output_csv)

def generate_multi_group(
    output_dir,
    root_audio_dir=audio_dir,
    root_segment_dir=segment_dir,
    first_year=2013,
    last_year=2018,
    middle=True,
    concert=True,
    symphonic=True,
    instruments=None,
    segment_status_csv=segment_status_csv
):
    """
    This is the main function that users should call.
    Generate audio-segment pair metadata for multiple (year, band, instrument) groups.
    Write the results into a .csv file in the output directory.

    Parameters
    ----------
    output_dir : str
        directory to output the metadata
    root_audio_dir : Optional[str]
        root directory of the audio files. Default: audio_dir (in the default config file)
    root_segment_dir : Optional[str]
        root directory of the manually labeled segmentation files. Default: segment_dir (in the default config file)
    first_year : Optional[int]
    last_year : Optional[int]
        generate metadata from first year to last year. Default: 2013 and 2018
    middle : Optional[bool]
    concert : Optional[bool]
    symphonic : Optional[bool]
        whether to generate the metadata of middle/concert/symphonic group. Default: True
    instruments : Optional[List[str]]
        which instruments to include. If None, include all instruments. Default: None
    segment_status_csv : Optional[str]
        path to the csv metadata that contains segmentation status for the data. Default: 

    Return
    ----------
    None
    """
    df = pd.read_csv(segment_status_csv)

    years = range(first_year, last_year + 1)
    bands = []
    if middle:
        bands.append("middle")
    if concert:
        bands.append("concert")
    if symphonic:
        bands.append("symphonic")
    if instruments is None:
        instruments = [
            "Bb Clarinet",
            "Trumpet",
            "Tenor Saxophone",
            "Flute",
            "Alto Saxophone",
            "French Horn", 
            "Oboe", 
            "Snare Drum",
            "Xylophone",
            "impani"
        ]

    yearbands = list(product(years, bands, instruments))
    
    for year, band, instrument in tqdm(yearbands):
        _generate(
            root_audio_dir,
            root_segment_dir,
            output_dir,
            year,
            band,
            instrument,
            df[(df["Year"] == year) & (df["Band"] == band) & (df["Instrument"] == instrument)]
        )


if __name__ == "__main__":
    import fire

    fire.Fire(generate_multi_group)
