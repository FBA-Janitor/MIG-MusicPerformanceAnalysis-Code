from curses.ascii import isblank
from itertools import product
import json
import os
from pprint import pprint
from typing import OrderedDict
import pandas as pd

from assessments_scores.organize_assessment import read_normalized_csv

from utils.default_configs_path import (
    audio_config_path,
    data_repo,
    audio_name_change_config_path,
)
from utils.utils import read_yaml_to_dict

import glob
import re

from tqdm import tqdm


def filename_to_id_check(filename):
    _, basename = os.path.split(filename)

    out = re.match(r"(\d{5})(.*)?\.mp3", basename)

    sid, suffix = out.groups()

    valid = (len(suffix) == 0)

    return int(sid), valid


def process(
    root,
    year,
    band,
    data_repo=data_repo,
    audio_config_path=audio_config_path,
    name_change_config_path=audio_name_change_config_path,
):

    name_change_config = read_yaml_to_dict(root, name_change_config_path)["filenames"]

    name_change_config = [
        config
        for config in name_change_config
        if config["Year"] == year and config["BandLevel"] == band
    ]

    assert len(name_change_config) in [0, 1]

    if len(name_change_config) == 1:
        error_list = name_change_config[0]["ErrorList"]
        sid_error_list = [e["StudentID"] for e in error_list]
    else:
        error_list = []
        sid_error_list = []

    audio_path = read_yaml_to_dict(root, audio_config_path)["audio"][year][band]

    files = sorted(glob.glob(os.path.join(root, audio_path, "**/*.mp3"), recursive=True))

    audio_folder = os.path.join(root, data_repo, "cleaned", "audio", "bystudent", str(year), band)
    os.makedirs(audio_folder, exist_ok=True)

    symlink_path = os.path.join(audio_folder, "{sid}", "{sid}.mp3")

    for file in tqdm(files):
        sid, valid = filename_to_id_check(file)

        if sid in sid_error_list:
            error_entry = error_list[sid_error_list.index(sid)]

            fstem = file.replace(root + os.path.sep, "")
            if error_entry['OldFileName'] == fstem:
                assert not valid
                if error_entry.get('Reason', None) == 'Duplicate':
                    print('Duplicate, skipping: ', file)
                    continue
                else:
                    print(error_entry.get('Reason', "no reason provided"))
        else:
            assert valid

        os.makedirs(os.path.join(audio_folder, str(sid)), exist_ok=True)

        audio_path = symlink_path.format(sid=sid)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        os.symlink(file, audio_path)


def process_multiyear(
    root: str,
    first_year=2013,
    last_year=2018,
    middle=True,
    concert=True,
    symphonic=True,
    data_repo=data_repo,
    audio_config_path=audio_config_path,
    name_change_config_path=audio_name_change_config_path,
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

    out = []

    for year, band in tqdm(yearbands):
        process(
            root,
            year,
            band,
            data_repo=data_repo,
            audio_config_path=audio_config_path,
            name_change_config_path=name_change_config_path,
        )


if __name__ == "__main__":
    import fire

    fire.Fire()
