import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from core.subdatasets import GenericSubdataset

class SegmentDataset(GenericSubdataset):
    def __init__(
        self,
        student_information : List[Tuple],
        data_root : str
    ) -> None:
        super().__init__(student_information=student_information, data_root=data_root)

    def _load_data_path(self):

        for (sid, year, band) in self.student_information:

            segment_path = os.path.join(self.data_root, str(year), band, "{}/{}_seginst.csv".format(sid, sid))
            if not os.path.exists(segment_path):
                warnings.warn("Missing segment file: {}".format(segment_path))
                continue

            self.data_path[sid] = segment_path

    def read_data_file(self, segment_path):
        if segment_path is None:
            return np.zeros([5, 2])
        seg_df = pd.read_csv(segment_path)
        start = seg_df["Start"]
        end = seg_df["End"]
        return np.vstack([start, end]).T
