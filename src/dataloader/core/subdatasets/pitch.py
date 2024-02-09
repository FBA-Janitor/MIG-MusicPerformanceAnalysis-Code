import os
from . import GenericSubdataset
from typing import List, Tuple
import pandas as pd
import numpy as np


class PitchDataset(GenericSubdataset):
    def __init__(
        self,
        student_information: List[Tuple],
        target_length_second=None,
        data_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack/bystudent",
        to_midi=True,
        normalize_mean=36,
        normalize_std=72,
        hop_size_second=None,
        oldheader=None,
        filename_format=None,
        confidence_thresh=-1.0,
    ) -> None:
        
        if oldheader is None:
            if "pitchtrack/bystudent" in data_root:
                # old style header ("MIDI" instead of "freq" even though it's in Hz)
                oldheader = True
            elif "pitchtrack3/bystudent" in data_root:
                # new style header
                oldheader = False
            else:
                raise ValueError("Cannot determine header from data_root")

        self.oldheader = oldheader

        if filename_format is None:
            if "pitchtrack/bystudent" in data_root:
                filename_format = "{sid}_pyin_pitchtrack.csv"
            elif "pitchtrack3/bystudent" in data_root:
                filename_format = "{sid}.f0.csv"
            else:
                raise ValueError("Cannot determine filename_format from data_root")

        self.filename_format = filename_format

        self.confidence_thresh = confidence_thresh

        self.to_midi = to_midi
        if hop_size_second is None:
            if "pitchtrack/bystudent" in data_root:
                hop_size_second = 256 / 44100 # approx 5.8 milliseconds
            elif "pitchtrack3/bystudent" in data_root:
                hop_size_second = 10 * 1e-3 # 10 milliseconds
            else:
                raise ValueError("Cannot determine hop_size_second from data_root")
        
        self.target_length_second = target_length_second if (target_length_second is not None and target_length_second > 0) else None
        self.target_length_frames = (
            int(np.ceil(self.target_length_second / hop_size_second) + 1)
            if self.target_length_second is not None
            else None
        )
        
        self.normalize_mean = normalize_mean if normalize_mean is not None else 0.0
        self.normalize_std = normalize_std if normalize_std is not None else 1.0

        super().__init__(student_information=student_information, data_root=data_root)

    def _load_data_path(self):
        """
        Overwrite this in the subclass
        Load the path to the data into self.path
        """
        found = 0

        for (sid, year, band) in self.student_information:

            f0_path = os.path.join(
                self.data_root, str(year), band, str(sid), self.filename_format.format(sid=sid)
            )

            if os.path.exists(f0_path):
                found += 1
                self.data_path[str(sid)] = f0_path

        print(
            f"Requested {len(self.student_information)} students: {found} have usable pitch data."
        )

    def validated_student_information(self):
        return [x for x in self.student_information if str(x[0]) in self.data_path]

    def read_data_file(self, data_path, start, end, segment=None):
        """
        Overwrite this in the subclass
        Read the data by the path
        """

        df = pd.read_csv(data_path)

        if self.oldheader:
            df = df.rename(columns={"MIDI": "frequency", "Time": "time"})

        time = df["time"].values.squeeze()
        f0 = df["frequency"].values.squeeze()

        if self.oldheader:
            confidence = np.ones_like(f0)
            bool_masks = (f0 != 0)
        else:
            confidence = df["confidence"].values.squeeze()
            bool_masks = (confidence > self.confidence_thresh)

        # f0[~bool_masks] = np.nan

        if self.to_midi:
            # f0 = np.log2(f0 / 440) * 12 + 69
            f0[f0 != 0] = 69 + 12 * np.log2(f0[f0 != 0] / 440)
            f0[f0 != 0] = (f0[f0 != 0] - self.normalize_mean) / self.normalize_std

        if start is not None:
            time_filt = time >= start
        else:
            time_filt = np.ones_like(time, dtype=bool)

        if end is not None:
            time_filt = time_filt & (time <= end)

        f0 = f0[time_filt]
        confidence = confidence[time_filt]
        masks = bool_masks[time_filt]

        if self.target_length_frames is not None and f0.shape[0] != self.target_length_frames:
            nf0 = f0.shape[0]
            if nf0 > self.target_length_frames:
                raise ValueError(
                    f"Length of f0 is {nf0}, which is longer than the maximum length {self.target_length_frames}"
                )

            def pad(x):
                return np.pad(
                    x,
                    (0, self.target_length_frames - nf0),
                    mode="constant",
                    constant_values=0.0,
                )

            f0 = pad(f0)
            masks = pad(masks)
            confidence = pad(confidence)
        else:
            pass

        # f0 = np.nan_to_num(f0, nan=0.0, posinf=128.0, neginf=0.0)

        return {
            "f0": f0.astype(np.float32),
            "mask": masks,
            "confidence": confidence.astype(np.float32),
        }


if __name__ == "__main__":

    f0ds = PitchDataset(
        student_information=[(29522, 2013, "concert"), (30349, 2013, "concert")]
    )

    for m in f0ds.get_item_by_student_id(29522):
        print(m)