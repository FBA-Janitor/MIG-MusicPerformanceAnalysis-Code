from itertools import product
from typing import Optional
import pandas as pd

import os

from tqdm import tqdm

from utils.default_configs_path import data_repo, xlsx_config_path, missing_max_score_config_path, assessment_name_change_config_path
from utils.utils import read_yaml_to_dict

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
        xlsx_config = read_yaml_to_dict(root, xlsx_config_path)
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
    trial: int = 0,     # FIXME: 
):
    """
    Get a DataFrame containing maximum scores.

    Parameters
    ----------
    year : int
        year of the data
    band : str
        group of the band, 'middle', 'concert' or 'symphonic'
    score_grp: str
        name of score group
    desc: str
        description
    instrument: str
        instrument
    max_score_df: pd.DataFrame
        DataFrame with the maximum scores
    missing_max_score_config
    Returns
    -------
    pd.DataFrame
        DataFrame with the maximum scores
    """

    if trial > 1:
        print(year, band, score_grp, desc)
        print(f"Trial: {trial}")
        raise RecursionError

    df = max_score_df

    isperc = instrument == "Percussion"

    if isperc:
        df = df[df["Percussion"] == 1.0]
    else:
        df = df[df["Percussion"].isna()]

    schoolyear = f"{year}-{year+1}"

    df = df[df["SchoolYear"] == schoolyear]
    df = df[df["ScoreGroup"] == score_grp]
    df = df[df["Description"] == desc]

    if len(df["MaxScore"].to_list()) == 0:
        try:
            for replacement_dict in missing_max_score_config["missing"]:

                conditions = replacement_dict["Conditions"]

                if year not in conditions.get("Year", [year]):
                    continue

                if band not in conditions.get("BandLevel", [band]):
                    continue

                if score_grp not in conditions.get("ScoreGroup", [score_grp]):
                    continue

                if desc not in conditions.get("Description", [desc]):
                    continue

                replacement = replacement_dict["ReplaceWith"]

                score_grp = replacement.get("ScoreGroup", score_grp)
                desc = replacement.get("Description", desc)
                instrument = replacement.get("Instrument", instrument)
                year = replacement.get("Year", year)
                band = replacement.get("BandLevel", band)

                break

            return get_max_score_from_df(
                year,
                band,
                score_grp,
                desc,
                instrument,
                max_score_df,
                missing_max_score_config,
                trial=trial + 1,
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
    """
    Read xlsx file of raw assessment scores

    Parameters
    ----------
    root : str
        root directory of the FBA project
    year : int
        year of the data to be read
    band : str
        group of the band, 'middle', 'concert' or 'symphonic'
    data_repo : str, optional
        relative path to the repository housing cleaned data, by default data_repo
    xlsx_config_path : str, optional
        path to configuration file with locations of the excel assessment files, by default xlsx_config_path
    write_csv: bool, optional
        whether to write the content to a csv copy, by default False
    
    Returns
    -------
    pd.DataFrame
        DataFrame with raw assessment scores
    """

    xlsx_config = read_yaml_to_dict(root, xlsx_config_path)

    df = pd.read_excel(
        os.path.join(root, xlsx_config["assessment_scores"][year][band]["excel"]),
        sheet_name=xlsx_config["assessment_scores"][year][band]["sheet"],
    )

    if write_csv:
        folder = os.path.join(root, data_repo, "cleaned", "assessment", "summary")
        os.makedirs(folder, exist_ok=True)
        print("Writing raw csv:", os.path.join(folder, f"{year}_{band}_raw.csv"))
        df.to_csv(
            os.path.join(folder, f"{year}_{band}_raw.csv"), index=False, header=True
        )

    return df


def column_name_update(
    root: str,
    df: pd.DataFrame,
    year: int,
    band: str,
    name_change_config_path: str = assessment_name_change_config_path,
):
    """
    Update the column name of the score DataFrame

    Parameters
    ----------
    root : str
        root directory of the FBA project
    df : pd.DataFrame
        DataFrame of the scores
    year : int
        data of which year to be processed
    band : str
        group of the band, 'middle', 'concert' or 'symphonic'
    name_change_config_path: str, optional
        path to configuration file with name change in assessment, by default assessment_name_change_config_path

    Returns
    -------
    pd.DataFrame
        DataFrame with column name updated
    """

    name_change_config = read_yaml_to_dict(root, name_change_config_path)
    column_name_changes = name_change_config["columns"].get(year, {}).get(band, None)
    if column_name_changes:
        df = df.rename(columns=column_name_changes)
    return df


def instrument_name_update(
    root: str,
    df: pd.DataFrame,
    year: int,
    band: str,
    name_change_config_path: str = assessment_name_change_config_path,
):
    """
    Update the instrument name of the score DataFrame

    Parameters
    ----------
    root : str
        root directory of the FBA project
    df : pd.DataFrame
        DataFrame of the scores
    year : int
        data of which year to be processed
    band : str
        group of the band, 'middle', 'concert' or 'symphonic'
    name_change_config_path: str, optional
        path to configuration file with name change in assessment, by default assessment_name_change_config_path
    
    Returns
    -------
    pd.DataFrame
        DataFrame with column name updated
    """

    inst_name_change_config = read_yaml_to_dict(root, name_change_config_path)[
        "instrument"
    ]

    for oldname, newname in inst_name_change_config.items():
        df.loc[df["Instrument"] == oldname, "Instrument"] = newname

    return df


def assessment_name_update(
    root: str,
    df: pd.DataFrame,
    year: int,
    band: str,
    name_change_config_path: str = assessment_name_change_config_path,
    drop_legacy: bool = True,
):
    """
    Update the assessment name of the score DataFrame

    Parameters
    ----------
    root : str
        root directory of the FBA project
    df : pd.DataFrame
        DataFrame of the scores
    year : int
        data of which year to be processed
    band : str
        group of the band, 'middle', 'concert' or 'symphonic'
    name_change_config_path: str, optional
        path to configuration file with name change in assessment, by default assessment_name_change_config_path
    drop_legacy: bool, optional
        whether to include the assessment change with reason Legacy, by default True
    Returns
    -------
    pd.DataFrame
        DataFrame with assessment name updated
    """

    name_change_config = read_yaml_to_dict(root, name_change_config_path)
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
    name_change_config_path: str = assessment_name_change_config_path,
    update_column_name: bool = True,
    update_assessment_name: bool = True,
    update_instrument_name: bool = True,
    drop_legacy: bool = True,
    normalized: bool = False,
):
    """
    Read in the summary csv file

    Parameters
    ----------
    root : str
        root directory of the FBA project
    year : int
        data of which year to be processed
    band : str
        group of the band, 'middle', 'concert' or 'symphonic'
    data_repo : str, optional
        relative path to the repository housing cleaned data, by default data_repo
    name_change_config_path: str, optional
        path to configuration file with name change in assessment, by default assessment_name_change_config_path,
    update_column_name: bool, optional
        whether to update wrong column names with correct ones in the DataFrame, by default True 
    update_assessment_name: bool, optional
        whether to update wrong assessment names with correct ones in the DataFrame, by default True
    update_instrument_name : bool, optional
        whether to update wrong instrument names with correct ones in the DataFrame, by default True
    drop_legacy: bool, optional
        whether to include the assessment change with reason Legacy, by default True
    normalized : bool, optional
        whether read the normalized score (instead of raw score), by default False
    Returns
    -------
    pd.DataFrame
        the score DataFrame
    """
    
    suffix = "normalized" if normalized else "raw"
    summarypath = os.path.join(root, data_repo, "cleaned", "assessment", "summary")
    df = pd.read_csv(os.path.join(summarypath, f"{year}_{band}_{suffix}.csv"))

    if update_column_name:
        df = column_name_update(root, df, year, band, name_change_config_path)

    if update_assessment_name:
        df = assessment_name_update(
            root, df, year, band, name_change_config_path, drop_legacy=drop_legacy
        )

    if update_instrument_name:
        df = instrument_name_update(root, df, year, band, name_change_config_path)

    return df


def read_normalized_csv(
    root: str,
    year: int,
    band: str,
    data_repo: str = data_repo,
):
    return read_summary_csv(
        root=root,
        year=year,
        band=band,
        data_repo=data_repo,
        name_change_config_path=assessment_name_change_config_path,
        update_column_name=False,
        update_assessment_name=False,
        update_instrument_name=False,
        drop_legacy=False,
        normalized=True,
    )


def normalize_summary_csv(
    root: str,
    year: int,
    band: str,
    update_assessment_name: bool = True,
    update_instrument_name: bool = True,
    drop_legacy: bool = True,
    data_repo: str = data_repo,
    xlsx_config_path: str = xlsx_config_path,
    missing_max_score_config_path: str = missing_max_score_config_path,
    name_change_config_path: str = assessment_name_change_config_path,
    write_csv: Optional[str] = None,
):
    """
    Read in normalized summary csv file

    Parameters
    ----------
    root : str
        root directory of the FBA project
    year : int
        data of which year to be processed
    band : str
        group of the band, 'middle', 'concert' or 'symphonic'
    update_assessment_name: bool, optional
        whether to update wrong assessment names with correct ones in the DataFrame, by default True
    update_instrument_name : bool, optional
        whether to update wrong instrument names with correct ones in the DataFrame, by default True
    drop_legacy: bool, optional
        whether to include the assessment change with reason Legacy, by default True
    write_summary_csv : bool, optional
        whether to write the content to a csv copy, by default True
    write_normalized_csv: bool, optional
        whether to write the content to a csv copy, by default True
    data_repo : str, optional
        relative path to the repository housing cleaned data, by default data_repo
    xlsx_config_path : str, optional
        path to configuration file with locations of the excel assessment files, by default xlsx_config_path
    missing_max_score_config_path: str, optional
        path to configuration file with name change in max score, by default missing_max_score_config_path,
    name_change_config_path: str, optional
        path to configuration file with name change in assessment, by default assessment_name_change_config_path,
    write_csv: Optional[str], optional  # FIXME: the variable type is confusing here, this should be a boolean variable
        path to write the normalized csv, by default {year}_{band}_normalized.csv
    Returns
    -------
    None
    """

    df = read_summary_csv(
        root,
        year,
        band,
        data_repo,
        name_change_config_path,
        update_column_name=True,
        update_assessment_name=False,
        update_instrument_name=False,
        drop_legacy=False,
        normalized=False
    )

    max_score_df = get_max_score_df(
        root,
        data_repo=data_repo,
        xlsx_config_path=xlsx_config_path,
        write_csv=False,
        read_csv=None,
    )
    missing_max_score_config = read_yaml_to_dict(root, missing_max_score_config_path)

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

    if update_assessment_name:
        df = assessment_name_update(
            root, df, year, band, name_change_config_path, drop_legacy=drop_legacy
        )

    if update_instrument_name:
        df = instrument_name_update(root, df, year, band, name_change_config_path)

    if write_csv == False:
        return df

    if write_csv is not None:   # FIXME: write_csv should be a boolean variable
        if write_csv == True:
            write_csv = os.path.join(
                root,
                data_repo,
                "cleaned",
                "assessment",
                "summary",
                f"{year}_{band}_normalized.csv",
            )

        assert type(write_csv) == str
        df.to_csv(write_csv, index=False, header=True)

    return df


def process(
    root: str,
    year: int,
    band: str,
    update_assessment_name: bool = True,
    update_instrument_name: bool = True,
    drop_legacy: bool = True,
    write_summary_csv: bool = True,
    write_normalized_csv: bool = True,
    data_repo: str = data_repo,
    xlsx_config_path: str = xlsx_config_path,
    missing_max_score_config_path: str = missing_max_score_config_path,
    name_change_config_path: str = assessment_name_change_config_path,
):
    """
    TODO: add documentation

    Parameters
    ----------
    root : str
        root directory of the FBA project
    year : int
        data of which year to be processed
    band : str
        group of the band, 'middle', 'concert' or 'symphonic'
    update_assessment_name: bool, optional
        whether to update wrong assessment names with correct ones in the DataFrame, by default True
    update_instrument_name : bool, optional
        whether to update wrong instrument names with correct ones in the DataFrame, by default True
    drop_legacy: bool, optional
        whether to include the assessment change with reason Legacy, by default True
    write_summary_csv : bool, optional
        whether to write the content to a csv copy, by default True
    write_normalized_csv: bool, optional
        whether to write the content to a csv copy, by default True
    data_repo : str, optional
        relative path to the repository housing cleaned data, by default data_repo
    xlsx_config_path : str, optional
        path to configuration file with locations of the excel assessment files, by default xlsx_config_path
    missing_max_score_config_path: str, optional
        path to configuration file with name change in max score, by default missing_max_score_config_path,
    name_change_config_path: str, optional
        path to configuration file with name change in assessment, by default assessment_name_change_config_path,
    Returns
    -------
    None
    """
    parse_summary_sheet(
        root=root,
        year=year,
        band=band,
        data_repo=data_repo,
        xlsx_config_path=xlsx_config_path,
        write_csv=write_summary_csv,
    )

    normalize_summary_csv(
        root=root,
        year=year,
        band=band,
        update_assessment_name=update_assessment_name,
        update_instrument_name=update_instrument_name,
        drop_legacy=drop_legacy,
        data_repo=data_repo,
        xlsx_config_path=xlsx_config_path,
        missing_max_score_config_path=missing_max_score_config_path,
        name_change_config_path=name_change_config_path,
        write_csv=write_normalized_csv,
    )


def process_multiyear(
    root: str,
    first_year=2013,
    last_year=2018,
    middle=True,
    concert=True,
    symphonic=True,
    update_assessment_name: bool = True,
    update_instrument_name: bool = True,
    drop_legacy: bool = True,
    write_summary_csv: bool = True,
    write_normalized_csv: bool = True,
    data_repo: str = data_repo,
    xlsx_config_path: str = xlsx_config_path,
    missing_max_score_config_path: str = missing_max_score_config_path,
    name_change_config_path: str = assessment_name_change_config_path,
):
    """
    TODO: add documentation

    Parameters
    ----------
    root : str
        root directory of the FBA project
    first_year : int, optional
        first year of data to clean (included), by default 2013
    last_year : int
        last year of data to clean (included), by default 2018
    middle: bool, optional
        whether to process the middle group, by default True
    concert: bool, optional
        whether to process the concert group, by default True
    symphonic: bool, optional
        whether to process the symphonic group, by default True
    update_assessment_name: bool, optional
        whether to update wrong assessment names with correct ones in the DataFrame, by default True
    update_instrument_name : bool, optional
        whether to update wrong instrument names with correct ones in the DataFrame, by default True
    drop_legacy: bool, optional
        whether to include the assessment change with reason Legacy, by default True
    write_summary_csv : bool, optional
        whether to write the content to a csv copy, by default True
    write_normalized_csv: bool, optional
        whether to write the content to a csv copy, by default True
    data_repo : str, optional
        relative path to the repository housing cleaned data, by default data_repo
    xlsx_config_path : str, optional
        path to configuration file with locations of the excel assessment files, by default xlsx_config_path
    missing_max_score_config_path: str, optional
        path to configuration file with name change in max score, by default missing_max_score_config_path,
    name_change_config_path: str, optional
        path to configuration file with name change in assessment, by default assessment_name_change_config_path,
    Returns
    -------
    None
    """

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
        process(
            root,
            year,
            band,
            update_assessment_name=update_assessment_name,
            update_instrument_name=update_instrument_name,
            drop_legacy=drop_legacy,
            write_summary_csv=write_summary_csv,
            write_normalized_csv=write_normalized_csv,
            data_repo=data_repo,
            xlsx_config_path=xlsx_config_path,
            missing_max_score_config_path=missing_max_score_config_path,
            name_change_config_path=name_change_config_path,
        )


if __name__ == "__main__":
    import fire

    fire.Fire()
