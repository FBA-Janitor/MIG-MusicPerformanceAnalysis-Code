import os
from . import GenericSubdataset
from typing import List, Tuple
import pandas as pd
import numpy as np

class PitchDataset(GenericSubdataset):
    def __init__(
        self,
        student_information: List[Tuple],
        max_length_second=None,
        data_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack/bystudent",
        to_midi=True,
        hop_size_second=256/44100,
    ) -> None:
        super().__init__(student_information=student_information, data_root=data_root)

        self.to_midi = to_midi
        self.max_length_frames = int(np.ceil(max_length_second/hop_size_second) + 1) if max_length_second is not None else None

    def _load_data_path(self):
        """
        Overwrite this in the subclass
        Load the path to the data into self.path
        """
        found = 0

        for (sid, year, band) in self.student_information:

            f0_path = os.path.join(
                self.data_root, str(year), band, f"{sid}/{sid}_pyin_pitchtrack.csv"
            )

            if os.path.exists(f0_path):
                found += 1
                self.data_path[str(sid)] = f0_path

        print(f"Requested {len(self.student_information)} students: {found} have usable pitch data.")

    def validated_student_information(self):
        return [x for x in self.student_information if str(x[0]) in self.data_path]

    def read_data_file(self, data_path, start, end, segment=None):
        """
        Overwrite this in the subclass
        Read the data by the path
        """

        df = pd.read_csv(data_path)
        
        time = df.Time.values.squeeze()

        # this is a misnomer, it's actually in Hz
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

        if self.max_length_frames is not None:
            nf0 = f0.shape[0]
            if nf0 > self.max_length_frames:
                raise ValueError(f"Length of f0 is {nf0}, which is longer than the maximum length {self.max_length_frames}")
            f0pad = np.pad(f0, (0, self.max_length_frames - nf0), mode="constant", constant_values=np.nan).astype(np.float32)
            masks = np.pad(np.ones_like(f0), (0, self.max_length_frames - nf0), mode="constant", constant_values=0).astype(bool)
        else:
            f0pad = f0.astype(np.float32)
            masks = np.ones_like(f0, dtype=bool)

        return {
            "f0": f0pad,
            "mask": masks,
        }

if __name__ == "__main__":

    f0ds = PitchDataset(
        student_information=[(29522, 2013, "concert"), (30349, 2013, "concert")]
    )

    for m in f0ds.get_item_by_student_id(29522):
        print(m)

