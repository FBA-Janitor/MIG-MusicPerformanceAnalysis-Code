import os
from itertools import product
from tqdm import tqdm

import pandas as pd

from utils.default_configs_path import (
    algo_segment_dir,
    summary_csv_dir
)


def arrange_group(
    result_dir,
    year,
    band,
    root_segment_dir=algo_segment_dir,
    summary_csv_dir=summary_csv_dir
):
    """
    Arrange the segment results

    Parameters
    ----------
    result_dir : str
        directory of the segmentation results
    year : int
        year
    band : str
        concert, middle or symphonic
    root_segment_dir : str, optional
        root directory to write the segmentation results, by default algo_segment_dir
    summary_csv_dir : str, optional
        path to the directory storing the summary csv files, by default summary_csv_dir
    """
    summary_csv = os.path.join(summary_csv_dir, "{}_{}_normalized.csv".format(year, band))
    df = pd.read_csv(summary_csv)
    df = df[["Student", "Instrument"]].drop_duplicates()
    
    stu_id_list = sorted(df.Student.unique())

    year_folder = os.path.join(root_segment_dir, 'bystudent', str(year))
    if not os.path.exists(year_folder):
        os.mkdir(year_folder)

    band_folder = os.path.join(year_folder, str(band))
    if not os.path.exists(band_folder):
        os.mkdir(band_folder)

    for stu_id in stu_id_list:
        result_file = os.path.join(result_dir, "{}.csv".format(stu_id))

        if not os.path.exists(result_file):
            print("Missing segmentation result: ", result_file)
            continue

        result = pd.read_csv(result_file)
        instrument = df[df["Student"] == stu_id]["Instrument"].item()

        if (not len(result.index) == 5) or (instrument == "Percussion"):
            print("Failed segmentation or percussion: ", result_file)
            continue
            
        result.insert(loc=0, column="ScoreGroup", value=["Lyrical Etude", "Technical Etude", "Chromatic Scale", "Major Scales", "Sight-Reading"])
        result.insert(loc=0, column="Instrument", value=[instrument] * 5)

        if not os.path.exists(os.path.join(band_folder, "{}".format(stu_id))):
            os.mkdir(os.path.join(band_folder, "{}".format(stu_id)))
        segment_file = os.path.join(band_folder, "{}/{}_seginst.csv".format(stu_id, stu_id))
        result.to_csv(segment_file, index=False)


if __name__ == '__main__':
    import fire

    fire.Fire()