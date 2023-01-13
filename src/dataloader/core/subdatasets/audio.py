import os
import warnings
from typing import List, Tuple

import librosa
import numpy as np

from core.subdatasets import GenericSubdataset
from core.preprocess.segment_feature import extract_segment_feature

class AudioDataset(GenericSubdataset):
    """
    A generic class for FBA Subdataset

    Parameters
    ----------
    student_information : List[Tuple]
        List of tuples (student_id, year, band), to load the data
    data_root : str
        root directory of the audio data
    sr : int
        sampling rate when read the audio
    """
    
    def __init__(
        self,
        student_information : List[Tuple],
        data_root : str,

        feature_save_dir : str,
        load_from_feature=True,
        sr=22050,
        block_size=4096,
        hop_size=2048
    ) -> None:
        
        self.feature_save_dir = feature_save_dir
        self.load_from_feature = load_from_feature
        self.sr = sr
        self.block_size = block_size
        self.hop_size = hop_size

        super().__init__(student_information=student_information, data_root=data_root)

    def _load_data_path(self):

        for (sid, year, band) in self.student_information:

            feature_path = os.path.join(self.feature_save_dir, "{}_{}_{}.npy".format(year, band, sid))
            audio_path = os.path.join(self.data_root, str(year), band, "{}/{}.mp3".format(sid, sid))

            self.data_path[sid] = (audio_path, feature_path)

    def read_data_file(self, data_path):
        audio_path, feature_path = data_path
        if self.load_from_feature:
            if not os.path.exists(feature_path):
                warnings.warn("Missing feature file: {}. Try audio file instead.".format(feature_path))
            else:
                return self._read_from_feature(feature_path)
        
        if not os.path.exists(audio_path):
            warnings.warn("Missing audio file: {}".format(audio_path))
            return None
        else:
            return self._read_from_audio(data_path)

    def _read_from_feature(self, feature_path):
        with open(feature_path, 'rb') as f:
            feature = np.load(f)
        return feature

    def _read_from_audio(self, data_path):
        # TODO: support flexible feature extraction
        audio_path, feature_path = data_path

        y, _ = librosa.load(audio_path, sr=self.sr)
        feature = extract_segment_feature(y, sr=self.sr, block_size=self.block_size, hop_size=self.hop_size)
        with open(feature_path, 'wb') as f:
            np.save(f, feature)
        return feature