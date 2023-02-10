import glob
import os
import re
from io import StringIO
from itertools import product
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import optimize

from utils.utils import read_yaml_to_dict
from utils.default_configs_path import (
    data_repo,
    segmentation_config_path,
    segmentation_code_config_path,
    segmentation_order_config_path,
    pyin_repo,
)

from assessment_scores.organize_scores import read_normalized_csv


def read_txt_to_df(txt_path, header="// data format: "):
    """
    Read the segmentation or instrument txt file into a DataFrame

    Parameters
    ----------
    txt_path : str
        path to the txt file
    header : str
        format of the header prefix

    Returns
    -------
    pd.DataFrame
    """

    with open(txt_path, "r") as f:
        content = f.read()

    if header not in content:
        if re.match(r"\d*", content):
            content = "start_time(sec)\tduration(sec)\n" + content
        else:
            content = "instrument name(str)\tactivity name(str)\n" + content

    content = content.replace(header, "")

    content = re.sub(r"[\t ]{2,}", "\t", content)

    df = pd.read_table(StringIO(content))

    if df.shape[-1] < 2:
        df = pd.read_table(StringIO(content), delim_whitespace=True)

    return df


def read_pyin(pyin_path, correct_timestamp=True, k=7):

    with open(pyin_path, "r") as f:
        content = f.read()

    content = re.sub(",", "\t", content)

    df = pd.read_table(StringIO(content), names=["time", "frequency"])

    if correct_timestamp:
        t = df["Time"].to_numpy()

        tl = t[t < 100]
        tg = t[t >= 100]

        ntl = tl.shape[0]
        nt = t.shape[0]
        ntg = tg.shape[0]

        assert ntl + ntg == nt

        dt0 = t[1] - t[0]

        def error(dt):
            # print(np.arange(ntl).shape, np.arange(ntl + 1, nt).shape, tl.shape, tg.shape)
            return np.mean(
                np.concatenate(
                    [
                        np.abs(tl - np.round(dt * np.arange(ntl), 3)),
                        np.abs(tg - np.round(dt * np.arange(ntl + 1, nt + 1), 2)[:ntg]),
                    ]
                )
            )

        opt = optimize.minimize_scalar(error, bounds=(dt0 - 0.0005, dt0 + 0.0005))

        dt = opt.x
        t = dt * np.arange(nt)

        df["Time"] = t

    return df


def clean_instrument(
    thefolder,
    sid,
    col_rename,
    has_instrument,
    inst_codes,
    segm_codes,
    perc_inst,
    summary_df,
):
    """
    Organize the instrument files of a certain year and a certain group

    Parameters
    ----------
    thefolder : str
        subfolder containing the segment files
    sid : int
        student id
    col_rename : dict
        column name changes
    has_instrument : bool
        whether the subfolder contains the instrument labels
    inst_codes : dict
        instrument code to names
    segm_codes : dict
        segmentation code to names
    perc_inst : List[str]
        list of percussion instruments
    summary_df : pd.DataFrame
        summary DataFrame
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with instrument labels
    Optional[str]
        Instrument
    """

    if has_instrument:
        insts_df = read_txt_to_df(
            os.path.join(thefolder, f"{sid}_instrument.txt")
        ).rename(columns=col_rename)

        try:
            insts_df["Instrument"] = insts_df["Instrument"].apply(
                lambda i: inst_codes[i]
            )
            insts_df["ScoreGroup"] = insts_df["ScoreGroup"].apply(
                lambda s: segm_codes[s]
            )
        except Exception as e:
            print(sid)
            print(insts_df.columns)
            print(insts_df)
            raise e

        instrument = insts_df["Instrument"].unique()

        if len(instrument) > 1:
            for i in instrument:
                assert i in perc_inst

            instrument = "Percussion"
            # print(insts_df["ScoreGroup"].to_list())
            # print(insts_df["Instrument"].to_list())
        else:
            instrument = instrument[0]
    else:
        instrument = None
        insts_df = None

    sdf_inst0 = summary_df[summary_df["Student"].astype(int) == int(sid)]
    if len(sdf_inst0) == 0:
        # print(int(sid) in summary_df["Student"].astype(int).unique().tolist())
        # print(sid)
        return None, None

    sdf_inst = sdf_inst0["Instrument"].unique()
    if len(sdf_inst) == 0:
        # print(sdf_inst0)
        return None, None

    assert len(sdf_inst) == 1, (sid, sdf_inst)
    sdf_inst = sdf_inst[0]

    if instrument is not None:
        assert sdf_inst == instrument
    else:
        instrument = sdf_inst

    return insts_df, instrument


def clean_segment(
    thefolder,
    sid,
    col_rename,
    has_instrument,
    inst_codes,
    segm_codes,
    perc_inst,
    segm_order,
    summary_df,
    remove_short_segment=True,
    short_segment_threshold=1.0,
):
    """
    Process a single segment file

    Parameters
    ----------
    thefolder : str
        subfolder containing the segment files
    sid : int
        student id
    col_rename : dict
        column name changes
    has_instrument : bool
        whether the subfolder contains the instrument labels
    inst_codes : dict
        instrument code to names
    segm_codes : dict
        segmentation code to names
    perc_inst : List[str]
        list of percussion instruments
    segm_order : List[dict]
        order of the segmentations in the audio
    summary_df : pd.DataFrame
        summary DataFrame
    remove_short_segment : bool
        whether to remove the short segments, by default True
    short_segment_threshold : float
        the threshold of short segment to remove, by default 1.0

    Returns
    -------
    Optional[dict]
    """

    segments = read_txt_to_df(os.path.join(thefolder, f"{sid}_segment.txt"))

    segments.columns = [c.strip() for c in segments.columns]
    segments = segments.rename(columns=col_rename)

    try:
        segments["End"] = segments["Start"] + segments["Duration"]
    except Exception as e:
        print(sid)
        print(segments.shape)
        print(segments)

    if remove_short_segment:
        segments = segments[
            segments["Duration"] > short_segment_threshold
        ].reset_index()

    insts_df, instrument = clean_instrument(
        thefolder,
        sid,
        col_rename,
        has_instrument,
        inst_codes,
        segm_codes,
        perc_inst,
        summary_df,
    )

    if insts_df is None and instrument is None:
        return None

    flag = {
        "Segment Length": False,
        "Instrument Length": False,
        "Segment-Instrument Agreement": False,
    }

    nseg = 7 if instrument == "Percussion" else 5

    wind_segments = [s["ScoreGroup"] for s in segm_order["winds"]]
    perc_segments = [s["ScoreGroup"] for s in segm_order["percussion"]]
    perc_insts = [s["Instrument"] for s in segm_order["percussion"]]

    if insts_df is not None:
        flag["Segment-Instrument Agreement"] = len(segments) != len(insts_df)
    flag["Segment Length"] = len(segments) != nseg
    flag["Instrument Length"] = len(insts_df) != nseg if insts_df is not None else False

    if np.any(list(flag.values())):
        #     print(flag)
        #     print(segments)
        #     print(insts_df)
        return {
            "segments": segments,
            "instruments": insts_df,
            "combined": None,
            "flags": flag,
        }

    if insts_df is not None:
        combined_df = pd.concat([insts_df, segments], axis=1)
    else:
        combined_df = segments.copy()

        if instrument == "Percussion":
            combined_df["Instrument"] = perc_insts
            combined_df["ScoreGroup"] = perc_segments
        else:
            combined_df["Instrument"] = instrument
            combined_df["ScoreGroup"] = wind_segments

    combined_df = combined_df[["Instrument", "ScoreGroup", "Start", "Duration", "End"]]

    return {
        "segments": segments,
        "instruments": insts_df,
        "combined": combined_df,
        "flags": flag,
    }


def process(
    root,
    year,
    band,
    data_repo=data_repo,
    pyin_repo=pyin_repo,
    segmentation_config_path=segmentation_config_path,
    segmentation_code_config_path=segmentation_code_config_path,
    segmentation_order_config_path=segmentation_order_config_path,
):
    """
    Organize the segmentation files of a certain year and a certain group

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
    pyin_repo : str, optional
        relative path to the repository housing cleaned pitch tracking data, by default pyin_repo
    segmentation_config_path : str, optional
        path to configuration file with locations of the segmentation files, by default segmentation_config_path
    segmentation_code_config_path: str, optional
        path to configuration file with the meanings of segmentation codes, by default segmentation_code_config_path
    segmentation_order_config_path: str, optional
        path to configuration with the order of segmentations, by default segmentation_order_config_path

    Returns
    -------
    None
    """

    summary_df = read_normalized_csv(root, year, band, data_repo)

    segmentation_path = read_yaml_to_dict(root, segmentation_config_path)[
        "segmentation"
    ][year][band]

    segmentation_code_config = read_yaml_to_dict(root, segmentation_code_config_path)
    inst_codes = segmentation_code_config["instrument"]
    segm_codes = segmentation_code_config["segment"]
    col_rename = segmentation_code_config["columns"]

    perc_inst = segmentation_code_config["percussion instruments"]

    segmentation_order_config = read_yaml_to_dict(root, segmentation_order_config_path)

    subfolders = [
        f
        for f in os.listdir(os.path.join(root, segmentation_path))
        if os.path.isdir(os.path.join(root, segmentation_path, f))
    ]

    not_found = []

    for sid in tqdm(subfolders):
        assert re.match(r"^\d{5}$", sid) is not None

        # thefolder is the path of subfolder
        thefolder = os.path.join(root, segmentation_path, sid)
        contents = os.listdir(thefolder)

        has_segment = f"{sid}_segment.txt" in contents
        has_instrument = f"{sid}_instrument.txt" in contents

        if has_segment:
            cleaned_segment = clean_segment(
                thefolder,
                sid,
                col_rename,
                has_instrument,
                inst_codes,
                segm_codes,
                perc_inst,
                segmentation_order_config,
                summary_df,
            )

            if cleaned_segment is None:
                not_found.append(int(sid))
                continue

            outpath = os.path.join(
                root,
                data_repo,
                "cleaned",
                "segmentation",
                "bystudent",
                str(year),
                band,
                str(sid),
            )
            os.makedirs(outpath, exist_ok=True)

            segm_path = os.path.join(outpath, f"{sid}_segment.txt")

            if os.path.exists(segm_path) or os.path.islink(segm_path):
                os.remove(segm_path)

            os.symlink(
                os.path.join(thefolder, f"{sid}_segment.txt"),
                segm_path,
            )

            if has_instrument:
                inst_path = os.path.join(outpath, f"{sid}_instrument.txt")

                if os.path.exists(inst_path) or os.path.islink(segm_path):
                    os.remove(inst_path)

                os.symlink(
                    os.path.join(thefolder, f"{sid}_instrument.txt"),
                    inst_path,
                )

            combined_df = cleaned_segment["combined"]

            if combined_df is not None:
                combined_df.to_csv(
                    os.path.join(outpath, f"{sid}_seginst.csv"), index=False
                )

        has_pitchtrack = f"{sid}_pyin_pitchtrack.txt" in contents

        if has_pitchtrack:
            pyin = read_pyin(os.path.join(thefolder, f"{sid}_pyin_pitchtrack.txt"))

            outpath = os.path.join(
                root,
                pyin_repo,
                "cleaned",
                "pitchtrack",
                "bystudent",
                str(year),
                band,
                str(sid),
            )
            os.makedirs(outpath, exist_ok=True)

            pyin.to_csv(
                os.path.join(outpath, f"{sid}_pyin_pitchtrack.csv"), index=False
            )

            pyin_path = os.path.join(outpath, f"{sid}_pyin_pitchtrack.txt")

            if os.path.exists(pyin_path) or os.path.islink(pyin_path):
                os.remove(pyin_path)

            os.symlink(
                os.path.join(thefolder, f"{sid}_pyin_pitchtrack.txt"),
                pyin_path,
            )

            # print(pyin)

    if len(not_found) == 0:
        return

    print(
        f"\n\n{len(not_found)}/{len(subfolders)} students not found in summary sheet."
    )

    has_audio = []
    no_audio = not_found.copy()

    for year in range(2013, 2018 + 1):
        for band in ["concert", "middle", "symphonic"]:

            summary_df = pd.read_csv(
                os.path.join(
                    root,
                    data_repo,
                    "cleaned",
                    "assessment",
                    "summary",
                    f"{year}_{band}_normalized.csv",
                )
            )

            student_list = [int(s) for s in summary_df["Student"].unique().tolist()]
            common = set(student_list).intersection(not_found)

            if len(common) > 0:
                print()
                print(f"Checking for {year} {band}")
                print(f"Found: {len(common)}")
                if len(common) < 10:
                    print(common)
                print(f"Number of students in summary sheet: {len(student_list)}")

            for nf in tqdm(no_audio):
                audio_files = glob.glob(
                    os.path.join(
                        root,
                        data_repo,
                        f"cleaned/audio/bystudent/{year}/{band}",
                        f"**/{sid}*.mp3",
                    ),
                    recursive=True,
                )
                # print(nf, audio_files)

                if len(audio_files) > 0:
                    print()
                    print(f"{nf} found in {year} {band}")
                    has_audio.append(nf)
                    no_audio.remove(nf)

    print()
    print(
        f"{len(has_audio)} have audio files somewhere. {no_audio} have no audio anywhere."
    )

    # print(not_found)


def process_multiyear(
    root: str,
    first_year=2013,
    last_year=2018,
    middle=True,
    concert=True,
    symphonic=True,
    data_repo=data_repo,
    pyin_repo=pyin_repo,
    segmentation_config_path=segmentation_config_path,
    segmentation_code_config_path=segmentation_code_config_path,
    segmentation_order_config_path=segmentation_order_config_path,
):
    """
    Organize the audio files of several years and several groups

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
    data_repo : str, optional
        relative path to the repository housing cleaned data, by default data_repo
    pyin_repo : str, optional
        relative path to the repository housing cleaned pitch tracking data, by default pyin_repo
    segmentation_config_path : str, optional
        path to configuration file with locations of the segmentation files, by default segmentation_config_path
    segmentation_code_config_path: str, optional
        path to configuration file with the meanings of segmentation codes, by default segmentation_code_config_path
    segmentation_order_config_path: str, optional
        path to configuration with the order of segmentations, by default segmentation_order_config_path

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
        print(year, band)
        process(
            root,
            year,
            band,
            data_repo=data_repo,
            pyin_repo=pyin_repo,
            segmentation_config_path=segmentation_config_path,
            segmentation_code_config_path=segmentation_code_config_path,
            segmentation_order_config_path=segmentation_order_config_path,
        )


if __name__ == "__main__":
    import fire

    fire.Fire()
