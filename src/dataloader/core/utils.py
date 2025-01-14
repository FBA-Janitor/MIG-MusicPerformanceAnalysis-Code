from enum import Enum
import os
from typing import List, Optional
import yaml
from itertools import product

import pandas as pd

from . import types


def read_yaml_to_dict(path: str) -> dict:
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def get_student_ids_from_constraints(initial_set=None, *args, **kwargs):
    # TODO: fill in

    # initial_set is used to limit the search space, setting to None search thru everything
    pass


def get_student_information_for_segmentation(
    path_summary_csv="/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/data_parse/config/segmentation/segment_status_230922.csv",
    first_year=2013,
    last_year=2018,
    bands=["middle", "concert", "symphonic"],  # FIXME: use better argument
    instruments=["Flute", "Oboe", "BbClarinet"],
):
    df = pd.read_csv(path_summary_csv)

    years = range(first_year, last_year + 1)

    student_information = []
    years_bands_instruments = list(product(years, bands, instruments))
    print(years_bands_instruments)

    for year, band, instrument in years_bands_instruments:
        df_small = df[
            (df["Year"] == year)
            & (df["Band"] == band)
            & (df["Instrument"] == instrument)
        ]

        df_small = df_small[df_small["has_instrument"] & df_small["has_segment"]]
        df_small = df_small[["StudentID", "Year", "Band"]]

        for sid, year, band in df_small.values.tolist():
            student_information.append((str(sid), int(year), band))
        # student_information.extend([tuple(information) for information in df_small.values.tolist()])

    return sorted(student_information)


def get_year_band_inst(
    year_filter: Optional[List[int]],
    band_filter: Optional[List[types.BandType]],
    inst_filter: Optional[List[types.InstrumentType]],
):

    if year_filter is None:
        year = types.YearRange
    else:
        year = year_filter

    if band_filter is None:
        band = types.get_valid_strings_from_literal(types.BandType)
    else:
        band = band_filter

    if inst_filter is None:
        inst = ["all"]
    else:
        inst = []

        for i in inst_filter:
            if i in types.get_valid_strings_from_literal(types.InstrumentType):
                inst.append(i)
            else:
                inst.extend(
                    types.get_valid_strings_from_literal(types.__dict__[i + "Type"])
                )

    return list(product(year, band, inst))
