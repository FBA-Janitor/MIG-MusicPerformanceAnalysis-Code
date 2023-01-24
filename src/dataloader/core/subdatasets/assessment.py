import os
from . import GenericSubdataset
from typing import List, Tuple
import pandas as pd
import numpy as np


class AssessmentDataset(GenericSubdataset):
    def __init__(
        self,
        student_information: List[Tuple],
        data_root="/media/fba/MIG-FBA-Data-Cleaning/cleaned/assessment/summary",
        use_normalized=True,
    ) -> None:
        super().__init__(student_information=student_information, data_root=data_root)

        assert use_normalized, "Only normalized data is available."

    def _load_data_path(self):
        """
        Overwrite this in the subclass
        Load the path to the data into self.path
        """
        self.year_band = set()

        for _, year, band in self.student_information:
            self.year_band.add((year, band))

        for year, band in self.year_band:

            self.data_csv = {
                (year, band): pd.read_csv(
                    os.path.join(self.data_root, f"{year}_{band}_normalized.csv")
                )
            }

        for (sid, year, band) in self.student_information:

            self.data_path[str(sid)] = (sid, year, band)

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

        df = (
            df[
                (df["Student"] == sid)
                & df["ScoreGroup"].apply(lambda s: s.replace(" ", "") == segment)
            ][["Description", "NormalizedScore"]]
            .reset_index(drop=True)
            .set_index("Description")
            .to_dict()["NormalizedScore"]
        )

        df = dict(sorted(df.items()))

        return df


if __name__ == "__main__":

    f0ds = AssessmentDataset(
        student_information=[(29522, 2013, "concert"), (30349, 2013, "concert")]
    )

    print(f0ds.get_item_by_student_id(29522, segment="LyricalEtude"))
