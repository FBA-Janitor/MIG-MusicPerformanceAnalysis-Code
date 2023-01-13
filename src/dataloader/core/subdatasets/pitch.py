import os
from . import GenericSubdataset
from typing import List, Tuple
import pandas as pd
import numpy as np

class PitchDataset(GenericSubdataset):
    def __init__(
        self,
        student_information: List[Tuple],
        data_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack/bystudent",
        to_midi=True,
    ) -> None:
        super().__init__(student_information=student_information, data_root=data_root)

        self.to_midi = to_midi

    def _load_data_path(self):
        """
        Overwrite this in the subclass
        Load the path to the data into self.path
        """

        for (sid, year, band) in self.student_information:

            f0_path = os.path.join(
                self.data_root, str(year), band, f"{sid}/{sid}_pyin_pitchtrack.csv"
            )

            self.data_path[str(sid)] = f0_path

    def read_data_file(self, data_path, start, end, segment=None):
        """
        Overwrite this in the subclass
        Read the data by the path
        """

        df = pd.read_csv(data_path)
        
        time = df.Time.values.squeeze()

        f0 = df.MIDI.values.squeeze()

        f0[f0 == 0] = np.nan

        if self.to_midi:
            f0 = np.log2(f0 / 440) * 12 + 69

        if start is not None:
            time_filt = time >= start
        else:
            time_filt = np.ones_like(time, dtype=bool)

        if end is not None:
            time_filt = time_filt & (time <= end)



        f0 = f0[time_filt]

        return f0

if __name__ == "__main__":

    f0ds = PitchDataset(
        student_information=[(29522, 2013, "concert"), (30349, 2013, "concert")]
    )

    for m in f0ds.get_item_by_student_id(29522):
        print(m)

