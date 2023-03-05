import os
import warnings
from typing import List, Tuple

import librosa
import numpy as np

from . import GenericSubdataset


class AudioDataset(GenericSubdataset):
    def __init__(
        self,
        student_information: List[Tuple],
        data_root: str,
        sr=22050,
    ) -> None:
        

        self.sr = sr
        super().__init__(student_information=student_information, data_root=data_root)

    def read_data_file(self, data_path, start=None, end=None, segment=None):
        return self.read_audio(data_path, start=start, end=end, segment=segment)

    def read_audio(self, data_path, start, end, segment=None):
                
        y, _ = librosa.load(data_path, sr=self.sr, offset=start, duration=end-start)

        return y

    def _load_data_path(self):

        found = 0


        for (sid, year, band) in self.student_information:
            audio_path = os.path.join(
                self.data_root, str(year), band, "{}/{}.wav".format(sid, sid)
            )

            if os.path.exists(audio_path):
                found += 1
                self.data_path[str(sid)] = audio_path

        print(f"Requested {len(self.student_information)} students: {found} have usable audio.")

    def validated_student_information(self):
        return [x for x in self.student_information if str(x[0]) in self.data_path]
