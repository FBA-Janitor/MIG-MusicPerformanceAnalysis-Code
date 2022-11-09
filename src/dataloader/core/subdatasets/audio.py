import os
import warnings
from typing import List, Tuple

import librosa

from core.subdatasets import GenericSubdataset

class AudioDataset(GenericSubdataset):
    def __init__(
        self,
        student_information : List[Tuple],
        data_root : str,
        sr=22050,
    ) -> None:
        super().__init__(student_information=student_information, data_root=data_root)
        
        self.sr = sr

    def _load_data_path(self):

        for (sid, year, band) in self.student_information:

            audio_path = os.path.join(self.data_root, str(year), band, "{}/{}.mp3".format(sid, sid))
            if not os.path.exists(audio_path):
                warnings.warn("Missing audio file: {}".format(audio_path))

            self.data_path[sid] = audio_path

    def read_data_file(self, audio_path):
        y, _ = librosa.load(audio_path, sr=self.sr)
        return y
