import os

import scipy.signal
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
        hop_size_second=None,
        oldheader=None,
        filename_format=None,
        confidence_thresh=0.0,
        augment=False,
        chunk_size_second=None,
        chunk_hop_size_second=None,
        target_length=None
    ) -> None:
        
        if oldheader is None:
            if "pitchtrack_pyin" in data_root:
                # old style header ("MIDI" instead of "freq" even though it's in Hz)
                oldheader = True
            else:
                # new style header
                oldheader = False

        self.oldheader = oldheader

        self.already_in_midi = False

        if filename_format is None:
            if "pitchtrack_pyin" in data_root:
                filename_format = "{sid}_pyin_pitchtrack.csv"
            elif "pitchtrack_pesto" in data_root:
                filename_format = "{sid}.csv"
                self.already_in_midi = True
            else:
                raise ValueError("Cannot determine filename_format from data_root")

        self.filename_format = filename_format

        self.confidence_thresh = confidence_thresh

        self.to_midi = to_midi
        if hop_size_second is None:
            if "pitchtrack_pyin" in data_root:
                hop_size_second = 256 / 44100 # approx 5.8 milliseconds
            elif "pitchtrack_pesto_5805us" in data_root:
                hop_size_second = 256/44100
            elif "pitchtrack_pesto" in data_root:
                hop_size_second = 10 * 1e-3
            else:
                raise ValueError("Cannot determine hop_size_second from data_root")
            
        self.hop_size_second = hop_size_second

        self.max_length_frames = (
            int(np.ceil(max_length_second / hop_size_second) + 1)
            if max_length_second is not None
            else None
        )

        self.chunk_size_second = chunk_size_second
        self.chunk_hop_size_second = chunk_hop_size_second

        if self.chunk_size_second is not None:
            self.chunk_size_frames = np.round(chunk_size_second / hop_size_second).astype(int)
            self.chunk_hop_size_frames = np.round(chunk_hop_size_second / hop_size_second).astype(int) if chunk_hop_size_second is not None else None
        else:
            self.chunk_size_frames = None
            self.chunk_hop_size_frames = None

        self.augment = augment

        if self.augment:
            print("\n\n!!!!AUGMENTATION IS ON!!!!\n\n")

        self.n_students = len(student_information)

        super().__init__(student_information=student_information, data_root=data_root)

        if target_length is not None:

            n_loop = int(np.ceil(target_length / len(self.student_information)))

            student_information = self.student_information * n_loop

        elif chunk_size_second is not None and chunk_hop_size_second is not None:

            self.data_path_to_sid = {
                data_path: sid for sid, data_path in self.data_path.items()
            }

            n_frames_per_students = {
                sid: self.get_frame_count(data_path) for sid, data_path in self.data_path.items()
            }

            n_chunks_per_students = {
                sid: int(np.ceil(n_frames / self.chunk_hop_size_frames)) for sid, n_frames in n_frames_per_students.items()
            }

            self.length = sum(n_chunks_per_students.values())

            student_information = [
                (sid, year, band)
                for sid, year, band in self.student_information 
                for _ in range(n_chunks_per_students[sid])
            ]

            self.current_chunk = {
                sid: 0 for sid, _, _ in self.student_information
            }

            self.max_chunk = {
                sid: n_chunks_per_students[sid] for sid, _, _ in self.student_information
            }

        # yes I am initializing twice. this is intended
        super().__init__(student_information=student_information, data_root=data_root)
    
        # print(self.length)
        print("Dataset length: ", self.length)

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

            # print(f0_path)

            if os.path.exists(f0_path):
                found += 1
                # print(f"Found {f0_path}")
                self.data_path[str(sid)] = f0_path

        print(
            f"Requested {len(self.student_information)} students: {found} have usable pitch data."
        )

    def validated_student_information(self):
        # print([x[0] for x in self.student_information])
        return [x for x in self.student_information if str(x[0]) in self.data_path]

    def get_frame_count(self, data_path):
        df = pd.read_csv(data_path)

        return len(df)

    def read_full_data_file(self, data_path):
        df = pd.read_csv(data_path)

        if self.oldheader:
            df = df.rename(columns={"MIDI": "frequency", "Time": "time"})
        

        time = df["time"].values.squeeze()
        f0 = df["frequency"].values.squeeze()

        if "confidence" not in df.columns:
            confidence = np.ones_like(f0)
            bool_masks = (f0 != 0)
        else:
            confidence = df["confidence"].values.squeeze()
            bool_masks = (confidence > self.confidence_thresh)

        f0[~bool_masks] = np.nan

        if self.to_midi and not self.already_in_midi:
            f0[f0 <= 0] = np.nan
            f0 = np.log2(f0 / 440) * 12 + 69

        return time, f0, confidence, bool_masks
    
    def segment_data_file(
            self,
            time,
            f0,
            confidence,
            bool_masks,
            start,
            end,
    ):
        
        if start is not None:
            time_filt = time >= start
        else:
            time_filt = np.ones_like(time, dtype=bool)

        if end is not None:
            time_filt = time_filt & (time <= end)

        f0 = f0[time_filt]
        confidence = confidence[time_filt]
        bool_masks = bool_masks[time_filt]

        return f0, confidence, bool_masks
    
    def chunk_data(
            self, 
            f0,
            confidence,
            bool_masks,
            data_path,
            size_scale=1.0
    ):
        
        if self.chunk_hop_size_frames is None:

            if f0.shape[0] - self.chunk_size_frames <= 0:
                start = 0
            else:
                start = np.random.randint(0, f0.shape[0] - self.chunk_size_frames)
        else:
            assert data_path is not None

            sid = self.data_path_to_sid[data_path]

            chunk_index = self.current_chunk[sid]
            self.current_chunk[sid] += 1
            if self.current_chunk[sid] >= self.max_chunk[sid]:
                self.current_chunk[sid] = 0

            start = chunk_index * self.chunk_hop_size_frames

        end = start + np.ceil(self.chunk_size_frames * size_scale).astype(int)

        f0 = f0[start:end]
        confidence = confidence[start:end]
        bool_masks = bool_masks[start:end]
    
        if f0.shape[0] < self.chunk_size_frames:
            f0 = np.pad(
                f0,
                (0, self.chunk_size_frames - f0.shape[0]),
                mode="constant",
                constant_values=0.0,
            )
            confidence = np.pad(
                confidence,
                (0, self.chunk_size_frames - confidence.shape[0]),
                mode="constant",
                constant_values=0.0,
            )
            bool_masks = np.pad(
                bool_masks,
                (0, self.chunk_size_frames - bool_masks.shape[0]),
                mode="constant",
                constant_values=False,
            )
        
        return f0, confidence, bool_masks
    
    def augment_data(
            self,
            f0,
            confidence,
            bool_masks,
            tempo_rate=100,
    ):
        
        if tempo_rate != 100:
            if tempo_rate > 100:
                up = tempo_rate
                down = 100
            else:
                up = 100
                down = tempo_rate

            f0 = scipy.signal.resample_poly(f0, up, down)
            confidence = scipy.signal.resample_poly(confidence, up, down)
            bool_masks = scipy.signal.resample_poly(bool_masks.astype(float), up, down).round().astype(bool)

        fshift = np.random.randint(-12, 12)
        f0 = f0 + fshift

        if f0.shape[0] > self.chunk_size_frames:
            f0 = f0[:self.chunk_size_frames]
            confidence = confidence[:self.chunk_size_frames]
            bool_masks = bool_masks[:self.chunk_size_frames]
        elif f0.shape[0] < self.chunk_size_frames:
            f0 = np.pad(
                f0,
                (0, self.chunk_size_frames - f0.shape[0]),
                mode="constant",
                constant_values=0.0,
            )
            confidence = np.pad(
                confidence,
                (0, self.chunk_size_frames - confidence.shape[0]),
                mode="constant",
                constant_values=0.0,
            )
            bool_masks = np.pad(
                bool_masks,
                (0, self.chunk_size_frames - bool_masks.shape[0]),
                mode="constant",
                constant_values=False,
            )

        
        return f0, confidence, bool_masks

    def pad_data(self, f0, confidence, bool_masks):
        nf0 = f0.shape[0]
        if nf0 > self.max_length_frames:
            raise ValueError(
                f"Length of f0 is {nf0}, which is longer than the maximum length {self.max_length_frames}"
            )

        def pad(x):
            return np.pad(
                x,
                (0, self.max_length_frames - nf0),
                mode="constant",
                constant_values=0.0,
            )

        f0 = pad(f0)
        masks = pad(bool_masks)
        pad_masks = pad(np.ones_like(bool_masks))
        confidence = pad(confidence)

        return f0, masks, pad_masks, confidence

    def read_data_file(self, data_path, start, end, segment=None):
        """
        Overwrite this in the subclass
        Read the data by the path
        """
        
        time, f0, confidence, bool_masks = self.read_full_data_file(data_path)

        f0, confidence, bool_masks = self.segment_data_file(
            time, f0, confidence, bool_masks, start, end
        )
        
        # if self.augment:
        #     tempo_rate = np.random.randint(80, 120)
        # else:
        #     tempo_rate = 100
        

        if self.chunk_size_frames is not None:
            f0, confidence, bool_masks = self.chunk_data(
                f0, confidence, bool_masks, data_path, size_scale=1.0 #/tempo_rate
            )
            # print("chunked", f0.shape)

        
        if self.augment:
            f0, confidence, bool_masks = self.augment_data(
                f0, confidence, bool_masks, tempo_rate=100 #tempo_rate
            )
            # print("augmented", f0.shape)


        if self.max_length_frames is not None and self.chunk_size_frames is None:
            f0, bool_masks, pad_masks, confidence = self.pad_data(f0, confidence, bool_masks)
        else:
            pad_masks = np.ones_like(bool_masks)

        f0 = np.nan_to_num(f0, nan=0.0, posinf=128.0, neginf=0.0)
            
        if self.chunk_size_frames is not None:
            assert f0.shape[0] == self.chunk_size_frames, (f0.shape, self.chunk_size_frames)
            assert confidence.shape[0] == self.chunk_size_frames
            assert bool_masks.shape[0] == self.chunk_size_frames
            assert pad_masks.shape[0] == self.chunk_size_frames

        return {
            "f0": f0.astype(np.float32),
            "mask": bool_masks,
            "pad_mask": pad_masks,
            "confidence": confidence.astype(np.float32),
        }


if __name__ == "__main__":

    f0ds = PitchDataset(
        student_information=[(29522, 2013, "concert"), (30349, 2013, "concert")]
    )

    for m in f0ds.get_item_by_student_id(29522):
        print(m)
