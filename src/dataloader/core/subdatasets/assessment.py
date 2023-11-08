import os

from ..types.segment import SegmentType
from . import GenericSubdataset
from typing import List, Tuple
import pandas as pd
import numpy as np


def abbreviate_description(description):

    description = description.lower()

    if "rhythm" in description:
        return "rhythm_accuracy"
    
    if "note" in description:
        return "note_accuracy"
    
    if "musicality" in description or "tempo" in description or "style" in description:
        return "musicality"
    
    if "tone" in description or "quality" in description:
        return "tone"
    
    raise ValueError(f"Unknown description {description}")


def quantize(x):

    if x < 0.2:
        return 0.1
    elif x < 0.4:
        return 0.25
    elif x <= 0.6:
        return 0.5
    elif x <= 0.8:
        return 0.75
    else:
        return 0.9


class AssessmentDataset(GenericSubdataset):
    def __init__(
        self,
        student_information: List[Tuple],
        segment: SegmentType,
        data_root="/media/fba/MIG-FBA-Data-Cleaning/cleaned/assessment/summary",
        assessments=None,
        use_normalized=True,
        output_format="array",
        quantized=True,
    ) -> None:
        assert use_normalized, "Only normalized data is available."
        self.use_normalized = use_normalized
        self.segment = segment

        if assessments is None:
            assessments = sorted(["musicality", "note", "rhythm", "tone"])
        else:
            assessments = sorted([abbreviate_description(a) for a in assessments])

        self.assessments = assessments

        self.output_format = output_format

        if quantized:
            self.quantize = quantize
        else:
            self.quantize = lambda x: x

        super().__init__(student_information=student_information, data_root=data_root)

    def _load_data_path(self):
        """
        Overwrite this in the subclass
        Load the path to the data into self.path
        """
        self.year_band = set()

        for _, year, band in self.student_information:
            self.year_band.add((year, band))

        print("Loading assessment data...", self.year_band)

        if self.use_normalized:
            self.data_csv = {
                (year, band): pd.read_csv(
                    os.path.join(self.data_root, f"{year}_{band}_normalized.csv")
                )
                for year, band in self.year_band
            }
        else:
            raise NotImplementedError

        self.data_path = {
            str(sid): (sid, year, band)
            for (sid, year, band) in self.student_information
        }

        print("Assessment data loaded.", self.data_csv.keys())

    def validated_student_information(self):

        validated_student_information = []

        for year, band in self.year_band:
            df = self.data_csv[(year, band)].copy()
            df = df[
                df["ScoreGroup"].apply(lambda s: s.replace(" ", "") == self.segment)
            ]

            df["Student"] = df["Student"].astype(int)

            score_count = df.value_counts("Student")
            # TODO: adjust this based on segment
            if self.segment not in ["TechnicalEtude", "LyricalEtude"]:
                raise NotImplementedError
            student_ids_in_df = score_count[score_count == 4].index.tolist()
            student_ids_in_info = [
                int(sid)
                for sid, y, b in self.student_information
                if y == year and b == band
            ]
            common_student_ids = set(student_ids_in_df).intersection(
                set(student_ids_in_info)
            )

            for sid_, _, _, in self.student_information:
                if int(sid_) in common_student_ids:
                    validated_student_information.append((sid_, year, band))

        return validated_student_information

    def get_item_by_student_id(self, sid, start=None, end=None, segment=None):
        return self.read_data_file(
            self.data_path[str(sid)], start=start, end=end, segment=segment
        )

    def read_data_file(self, data_path, start=None, end=None, segment=None):
        """
        Overwrite this in the subclass
        Read the data by the path

        """

        assert segment is not None, "Segment must be specified."

        sid, year, band = data_path

        df = self.data_csv[(year, band)]

        df["Student"] = df["Student"].astype(int)

        df = (
            df[
                (df["Student"] == int(sid))
                & df["ScoreGroup"].apply(lambda s: s.replace(" ", "") == segment)
            ][["Description", "NormalizedScore"]]
            .reset_index(drop=True)
        )

        # print(df.keys())

        df["Description"] = df["Description"].apply(abbreviate_description)

        df = (
            df.set_index("Description")
            .to_dict()["NormalizedScore"]
        )

        out = {
            description: self.quantize(df[description])
            for description in self.assessments
        }

        if self.output_format == "dict":
            return out  # musicality, note, rhythm, tone
        elif self.output_format == "array":
            # print(out.keys())
            return np.array([out[k] for k in self.assessments]).astype(np.float32)
        else:
            raise NotImplementedError

        # print(out)
        return out


if __name__ == "__main__":

    f0ds = AssessmentDataset(
        student_information=[(29522, 2013, "concert"), (30349, 2013, "concert")]
    )

    print(f0ds.get_item_by_student_id(29522, segment="LyricalEtude"))
