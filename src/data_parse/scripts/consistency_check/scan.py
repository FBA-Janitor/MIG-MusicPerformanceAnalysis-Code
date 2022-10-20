import glob
from itertools import product
from operator import sub
import re
import pandas as pd
import os

from utils.utils import read_yaml_to_dict
from utils import default_configs_path

from io import StringIO

from tqdm import tqdm

import numpy as np

from audio import organize_audio
from assessments_scores import organize_assessment


def process_audio(
    root, year, band, config_paths=default_configs_path, filelist: list = []
):
    task_list = organize_audio.get_task_list(
        root=root,
        year=year,
        band=band,
        audio_config_path=config_paths.audio_config_path,
        name_change_config_path=config_paths.audio_name_change_config_path,
    )

    for f, sid in task_list:

        filelist.append(
            {"sid": sid, "audio_file": f, "audio_year": year, "audio_band": band}
        )

    return filelist


def process_scores(
    root, year, band, config_paths=default_configs_path, filelist: list = []
):

    df = organize_assessment.read_normalized_csv(
        root, year, band, data_repo=config_paths.data_repo
    )

    # print(df)

    filelist += [
        {"sid": sid, "score_year": year, "score_band": band}
        for sid in df["Student"].unique()
    ]

    return filelist


def process_segmentation(root, year, band, config_paths=default_configs_path, filelist=[]):

    segmentation_path = read_yaml_to_dict(root, config_paths.segmentation_config_path)[
        "segmentation"
    ][year][band]

    subfolders = [
        f
        for f in os.listdir(os.path.join(root, segmentation_path))
        if os.path.isdir(os.path.join(root, segmentation_path, f))
    ]
    
    for sid in subfolders:
        assert re.match(r"^\d{5}$", sid) is not None
    
    filelist += [
        {"sid": f, "segmentation_year": year, "segmentation_band": band} for f in subfolders
    ]
    
    return filelist


def process_multiyear(
    root: str,
    first_year=2013,
    last_year=2018,
    middle=True,
    concert=True,
    symphonic=True,
    config_paths=default_configs_path,
    check_audio=True,
    check_scores=True,
    check_segmentation=True,
):
    years = range(first_year, last_year + 1)
    bands = []
    if middle:
        bands.append("middle")
    if concert:
        bands.append("concert")
    if symphonic:
        bands.append("symphonic")

    yearbands = list(product(years, bands))

    process_funcs = []

    if check_audio:
        process_funcs.append(("audio", process_audio))

    if check_scores:
        process_funcs.append(("scores", process_scores))

    if check_segmentation:
        process_funcs.append(("segmentation", process_segmentation))

    for name, func in process_funcs:
        print(f"Checking {name}")
        filelist = []
        for year, band in tqdm(yearbands):
            print(year, band)
            filelist = func(
                root, year, band, config_paths=config_paths, filelist=filelist
            )

        pd.DataFrame(filelist).to_csv(
            os.path.join(root, config_paths.tally_path, f"{name}.csv"), index=False
        )


if __name__ == "__main__":
    import fire

    fire.Fire()
