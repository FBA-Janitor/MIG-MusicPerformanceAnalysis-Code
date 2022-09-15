from asyncore import write
from typing import Optional
import pandas as pd
import numpy as np
import re

import os, shutil

import json

from tqdm import tqdm
from pprint import pprint

from collections import defaultdict

import yaml

xlsx_config_path = "./MIG-MusicPerformanceAnalysis-Code/src/data_parse/config/assessment_scores/assessment_score_paths.yaml"
missing_max_score_config_path = "./MIG-MusicPerformanceAnalysis-Code/src/data_parse/config/assessment_scores/max_score_replacements.yaml"
name_change_config_path = "./MIG-MusicPerformanceAnalysis-Code/src/data_parse/config/assessment_scores/assessment_score_name_changes.yaml"
data_repo = "MIG-FBA-Data-Cleaning"


def read_yaml_to_dict(root: str, path_relative_to_root: str) -> dict:
    """
    Read YAML file into a dictionary

    Parameters
    ----------
    root : str
        root directory of the FBA project
    path_relative_to_root : str
        path of the YAML file, relative to `root`

    Returns
    -------
    dict
        Dictionary content of the YAML file
    """
    with open(os.path.join(root, path_relative_to_root), "r") as stream:
        data = yaml.safe_load(stream)
    return data


def get_max_score_df(
    root: str,
    data_repo: str = data_repo,
    xlsx_config_path: str = xlsx_config_path,
    write_csv: bool = False,
    read_csv: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Get a DataFrame containing maximum scores.

    Parameters
    ----------
    root : str
        root directory of the FBA project
    data_repo : str, optional
        relative path to the repository housing cleaned data, by default data_repo
    xlsx_config_path : str, optional
        path to configuration file with locations of the excel assessment files, by default xlsx_config_path
    write_csv : bool, optional
        whether to write the content to a csv copy, by default False
    read_csv : Optional[bool], optional
        whether to read the data from the copied csv instead, by default None. If None, read from csv if the csv file exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with the maximum scores
    """

    csv_path = os.path.join(
        root, data_repo, "cleaned", "assessment", "summary", "max_scores.csv"
    )

    if read_csv is None:
        read_csv = os.path.exists(csv_path)

    if read_csv == True:
        assert os.path.exists(csv_path)
        return pd.read_csv(csv_path)
    else:
        xlsx_config = read_yaml_to_dict(xlsx_config_path)
        dfms = pd.read_excel(os.path.join(root, xlsx_config["max_scores"]))

        if write_csv:
            dfms.to_csv(csv_path, index=False, header=True)

    return dfms


def get_max_score_from_df(
    year: int,
    band: str,
    score_grp: str,
    desc: str,
    instrument: str,
    max_score_df: pd.DataFrame,
    missing_max_score_config: dict,
):

    df = max_score_df

    isperc = instrument == "Percussion"

    if isperc:
        df = df[df["Percussion"] == float(isperc)]
    else:
        df = df[df["Percussion"].isna()]

    schoolyear = f"{year}-{year+1}"

    df = df[df["SchoolYear"] == schoolyear]
    df = df[df["ScoreGroup"] == score_grp]
    df = df[df["Description"] == desc]

    if len(df["MaxScore"].to_list()) == 0:
        try:
            replacement = missing_max_score_config["missing"][year][band][score_grp][
                desc
            ]

            score_grp = replacement.get("ScoreGroup", score_grp)
            desc = replacement.get("Description", desc)
            instrument = replacement.get("Instrument", instrument)
            year = replacement.get("Year", year)
            band = replacement.get("BandLevel", band)

            return get_max_score_from_df(
                year,
                band,
                score_grp,
                desc,
                instrument,
                max_score_df,
                missing_max_score_config,
            )
        except KeyError as ke:
            print(year, band, score_grp, desc, instrument)
            raise ke

    return float(df["MaxScore"].to_list()[0])


def parse_summary_sheet(
    root: str,
    year: int,
    band: str,
    data_repo: str = data_repo,
    xlsx_config_path: str = xlsx_config_path,
    write_csv: bool = False,
):

    xlsx_config = read_yaml_to_dict(xlsx_config_path, root)

    df = pd.read_excel(
        os.path.join(root, xlsx_config["assessment_scores"][year][band]["excel"]),
        sheet_name=xlsx_config["assessment_scores"][year][band]["sheet"],
    )

    if write_csv:
        folder = os.path.join(root, data_repo, "cleaned", "assessment", "summary")
        os.makedirs(folder, exist_ok=True)
        df.to_csv(os.path.join(folder, f"{year}_{band}.csv"), index=False, header=True)

    return df


def column_name_update(
    root: str,
    df: pd.DataFrame,
    year: int,
    band: str,
    name_change_config_path: str = name_change_config_path,
):
    name_change_config = read_yaml_to_dict(name_change_config_path, root)
    column_name_changes = name_change_config["columns"].get(year, {}).get(band, None)
    if column_name_changes:
        df = df.rename(columns=column_name_changes)
    return df


def assessment_name_update(
    root: str,
    df: pd.DataFrame,
    year: int,
    band: str,
    name_change_config_path: str = name_change_config_path,
    drop_legacy: bool = True,
):
    name_change_config = read_yaml_to_dict(name_change_config_path, root)
    assessment_name_changes = name_change_config["assessments"]

    for name_change in assessment_name_changes:

        if (not drop_legacy) and name_change["Reason"] == "Legacy":
            continue

        process = False
        for affected in name_change["Affects"]:
            if affected["Year"] == year:
                if band in affected["BandLevel"]:
                    process = True
                    break

        if process:
            old_score_grp = name_change["OldScoreGroup"]
            old_desc = name_change.get("OldDescription", None)

            df_filt = df["ScoreGroup"] == old_score_grp
            if old_desc is not None:
                df_filt = df_filt & (df["Description"] == old_desc)

            if drop_legacy and name_change["Reason"] == "Legacy":
                df = df[~df_filt]
            else:
                new_score_grp = name_change.get("NewScoreGroup", None)
                new_desc = name_change.get("NewDescription", None)

                if new_score_grp is not None:
                    df.loc[df_filt, "ScoreGroup"] = new_score_grp

                if new_desc is not None:
                    df.loc[df_filt, "Description"] = new_desc

    return df


def read_summary_csv(
    root: str,
    year: int,
    band: str,
    data_repo: str = data_repo,
    name_change_config_path: str = name_change_config_path,
    update_column_name: bool = True,
    update_assessment_name: bool = True,
    drop_legacy: bool = True,
):
    summarypath = os.path.join(root, data_repo, "cleaned", "assessment", "summary")
    df = pd.read_csv(os.path.join(summarypath, f"{year}_{band}.csv"))

    if update_column_name:
        df = column_name_update(df, year, band, root, name_change_config_path)

    if update_assessment_name:
        df = assessment_name_update(
            df, year, band, root, name_change_config_path, drop_legacy=drop_legacy
        )

    return df


def normalize_summary_csv(
    root: str,
    year: int,
    band: str,
    data_repo: str = data_repo,
    xlsx_config_path: str = xlsx_config_path,
    missing_max_score_config_path: str = missing_max_score_config_path,
    name_change_config_path: str = name_change_config_path,
    write_csv: Optional[str] = None,
):

    df = read_summary_csv(
        year,
        band,
        root,
        data_repo,
        name_change_config_path,
        update_column_name=True,
        update_assessment_name=False,
    )

    max_score_df = get_max_score_df(
        xlsx_config_path, root, data_repo, write_csv=False, read_csv=None
    )
    missing_max_score_config = read_yaml_to_dict(missing_max_score_config_path, root)

    gb = df.groupby(["ScoreGroup", "Description", "Instrument"])

    for (score_grp, desc, instrument), indices in tqdm(gb.groups.items()):
        max_score = get_max_score_from_df(
            year,
            band,
            score_grp,
            desc,
            instrument,
            max_score_df,
            missing_max_score_config,
        )

        df.loc[indices, "MaxScore"] = max_score

    df["NormalizedScore"] = df["Score"].astype(float) / df["MaxScore"]

    df = assessment_name_update(df, year, band, root, name_change_config_path)

    if write_csv is not None:
        df.to_csv(write_csv, index=False, header=True)

    return df


if __name__ == "__main__":
    import fire

    fire.Fire()
