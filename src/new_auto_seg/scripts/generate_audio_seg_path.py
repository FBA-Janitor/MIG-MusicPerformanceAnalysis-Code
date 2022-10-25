import os
from itertools import product
from tqdm import tqdm

import pandas as pd

from utils.default_configs_path import (
    audio_dir,
    segment_dir,
    segment_status_csv
)

def write_audio_seg_csv(audio_seg, output_csv):
    f = open(output_csv, 'w')
    for stu_id, audio_file, segment_file in audio_seg:
        f.write("{},{},{}\n".format(stu_id, audio_file, segment_file))
    f.close()

def generate(
    root_audio_dir,
    root_segment_dir,
    output_dir,
    year,
    band,
    instrument,
    summary_df
):
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

    output_csv = os.path.join(output_dir, "{}_{}_{}_audio_seg.csv".format(year, band, instrument.replace(" ", "")))
    write_audio_seg_csv(id_audio_seg, output_csv)

def generate_multi_year(
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
    TODO add documentation
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
        generate(
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

    fire.Fire()
