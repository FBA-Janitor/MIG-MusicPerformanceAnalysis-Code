import os

from ..types.segment import SegmentType
from . import GenericSubdataset
from typing import List, Tuple
import pandas as pd
import numpy as np


class AssessmentDataset(GenericSubdataset):
    def __init__(
        self,
        student_information: List[Tuple],
        segment: SegmentType,
        data_root="/media/fba/MIG-FBA-Data-Cleaning/cleaned/assessment/summary",
        use_normalized=True,
        output_format="array",
    ) -> None:
        assert use_normalized, "Only normalized data is available."
        self.use_normalized = use_normalized
        self.segment = segment

        self.output_format = output_format

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
            student_ids_in_df = score_count[score_count % 4 == 0].index.tolist()
            student_ids_in_info = [
                int(sid)
                for sid, y, b in self.student_information
                if y == year and b == band
            ]
            common_student_ids = set(student_ids_in_df).intersection(
                set(student_ids_in_info)
            )

            validated_student_information.extend(
                [(sid, year, band) for sid in common_student_ids]
            )

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

        df = df[
                (df["Student"] == int(sid))
                & df["ScoreGroup"].apply(lambda s: s.replace(" ", "") == segment)
            ][["Description", "NormalizedScore"]].reset_index(drop=True)

        if len(df) == 4:
            df = df.set_index("Description").to_dict()["NormalizedScore"]
        else:
            df = df.groupby("Description")["NormalizedScore"].apply(list).to_dict()

        out = dict(sorted(df.items()))

        if self.output_format == "dict":
            return out  # musicality, note, rhythm, tone
        elif self.output_format == "array":
            # print(out.keys())
            return np.array(list(out.values())).astype(np.float32)
        else:
            raise NotImplementedError

        # print(out)
        return out


if __name__ == "__main__":

    f0ds = AssessmentDataset(
        student_information=[(29522, 2013, "concert"), (30349, 2013, "concert")]
    )

    print(f0ds.get_item_by_student_id(29522, segment="LyricalEtude"))