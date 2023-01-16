import os
from itertools import product
from tqdm import tqdm

import pandas as pd

from utils.default_configs_path import (
    algo_segment_dir,
    segment_status_csv
)

def arrange(
    result_dir,
    output_dir,
    summary_df
):
    stu_id_list = sorted(summary_df.StudentID.unique())

    for stu_id in stu_id_list:
        result_file = os.path.join(result_dir, "{}.csv".format(stu_id))

        if not os.path.exists(result_file):
            print("Missing segmentation result: ", result_file)
            continue

        result = pd.read_csv(result_file)
        
        if not len(result.index) == 5:
            print("Failed segmentation or percussion: ", result_file)
            continue
        
        result.insert(loc=0, column="ScoreGroup", value=["Lyrical Etude", "Technical Etude", "Chromatic Scale", "Major Scales", "Sight-Reading"])
        instrument = summary_df[summary_df["StudentID"] == stu_id]["Instrument"].item()
        result.insert(loc=0, column="Instrument", value=[instrument] * 5)

        if not os.path.exists(os.path.join(output_dir, "{}".format(stu_id))):
            os.mkdir(os.path.join(output_dir, "{}".format(stu_id)))
        segment_file = os.path.join(output_dir, "{}/{}_seginst.csv".format(stu_id, stu_id))
        result.to_csv(segment_file, index=False)


def arrange_multi_year(
    result_dir,
    root_segment_dir=algo_segment_dir,
    first_year=2013,
    last_year=2018,
    middle=True,
    concert=True,
    symphonic=True,
    segment_status_csv=segment_status_csv
):
    """
    TODO: add documentation
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

    yearbands = list(product(years, bands))

    for year, band in tqdm(yearbands):
        year_folder = os.path.join(root_segment_dir, 'bystudent', str(year))
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)

        band_folder = os.path.join(year_folder, str(band))
        if not os.path.exists(band_folder):
            os.mkdir(band_folder)
        
        arrange(
            result_dir,
            band_folder,
            df[(df["Year"] == year) & (df["Band"] == band)]
        )

if __name__ == '__main__':
    import fire

    fire.Fire()