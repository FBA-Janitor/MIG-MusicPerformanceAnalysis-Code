import os
from pprint import pprint
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import GenericSubdataset

FAILED = 0
SUCCESS = 1
PARTIAL = 2

class SegmentDataset(GenericSubdataset):
    """
    A generic class for FBA Subdataset

    Parameters
    ----------
    student_information : List[Tuple]
        List of tuples (student_id, year, band), to load the data
    data_root : str
        root directory of the data
    """
    
    def __init__(
        self,
        student_information : List[Tuple],
        data_root : str = "/media/fba/MIG-FBA-Data-Cleaning/cleaned/segmentation/bystudent",
        algo_data_root : str = "/media/fba/MIG-FBA-Segmentation/cleaned/algo-segmentation/bystudent",
    ) -> None:
        self.algo_data_root = algo_data_root
        super().__init__(student_information=student_information, data_root=data_root)

    def validated_student_information(self):
        return [x for x in self.student_information if str(x[0]) in self.data_path]

    def _load_data_path(self, verbose=False):

        requested = len(self.student_information)
        not_found = 0
        found = 0

        yearbands = set([(year, band) for sid, year, band in self.student_information])
        status = {}

        if self.algo_data_root is not None:
            for year, band in yearbands:
                algo_report = os.path.join(self.algo_data_root, "../summary", f"report_{year}_{band}.csv")
                dfr = pd.read_csv(algo_report)
                status[(year, band)] = dfr.set_index("StudentID").to_dict()["Status"]
            # pprint(status[(year, band)])

        for (sid, year, band) in self.student_information:

            segment_path = os.path.join(self.data_root, str(year), band, "{}/{}_seginst.csv".format(sid, sid))
            if os.path.exists(segment_path):
                found += 1
            else:
                if self.algo_data_root is not None:
                    if status[(year, band)].get(int(sid), None) == SUCCESS:
                        segment_path = os.path.join(self.algo_data_root, str(year), band, "{}/{}_seginst.csv".format(sid, sid))
                        if os.path.exists(segment_path):
                            found += 1
                        else:
                            not_found += 1
                            if verbose:
                                warnings.warn(f"No segment found for {sid}")
                            continue
                    else:
                        not_found += 1
                        if verbose:
                            warnings.warn(f"No segment found for {sid}")
                        continue
                else:
                    # continue
                    not_found += 1
                    continue

            self.data_path[str(sid)] = segment_path
            if verbose:
                print(self.data_path[str(sid)])
            verbose = False

        print(f"Requested {requested} students: {found} have usable segmentation; {not_found} do not.")

    def read_data_file(self, segment_path, **kwargs):
        seg_df = pd.read_csv(segment_path)
        start = seg_df["Start"]
        end = seg_df["End"]
        return np.vstack([start, end]).T

    def get_minimum_maximum_segment_length(self, segment_id):
        lengths = []
        for sid in self.student_ids:
            start, end = self.get_item_by_student_id(sid)[segment_id]
            lengths.append(end - start)
        return np.min(lengths), np.max(lengths)

if __name__ == "__main__":
    
    from ...torch.datamodule import FBADataModule
    ids = FBADataModule(
        [(2013, "middle", "Trumpet"), (2014, "middle", "BbClarinet")],{}
    ).train_ids

    # print(ids)

    segds = SegmentDataset(ids)

    for sid in segds.student_ids:
        print(segds.get_item_by_student_id(sid))