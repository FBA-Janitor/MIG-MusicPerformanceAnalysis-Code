import os
import warnings
from typing import List, Tuple

import librosa
import numpy as np

from . import GenericSubdataset
from ..preprocess.segment_feature import extract_segment_feature


class AudioBaseDataset(GenericSubdataset):
    def __init__(
        self,
        student_information: List[Tuple],
        data_root: str,
        sr=22050,
    ) -> None:
        super().__init__(student_information=student_information, data_root=data_root)

        self.sr = sr

    def read_data_file(self, data_path, start=None, end=None, segment=None):
        return self.read_audio(data_path, start=start, end=end, segment=segment)

    def read_audio(self, data_path, start, end, segment=None):
        audio_path = data_path
        y, _ = librosa.load(audio_path, sr=self.sr)

        if start is None:
            start = 0
        else:
            start = np.round(start * self.sr).astype(int)

        if end is not None:
            end = np.round(end * self.sr).astype(int) + 1

        return y[start:end]

    def _load_data_path(self):

        for (sid, year, band) in self.student_information:
            audio_path = os.path.join(
                self.data_root, str(year), band, "{}/{}.wav".format(sid, sid)
            )

            self.data_path[str(sid)] = audio_path


class AudioMelSpecDataset(AudioBaseDataset):
    def __init__(
        self,
        student_information: List[Tuple],
        data_root: str,
        sr=22050,
        n_fft=2048,
        hop_length=1024,
        n_mels=96,
    ) -> None:
        super().__init__(student_information, data_root, sr)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def read_data_file(self, data_path, start=None, end=None, segment=None):
        y = self.read_audio(data_path, start=start, end=end, segment=segment)

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        return np.transpose(mel_spec, (-1, -2)) # (time, freq)


class AudioFeatureDataset(AudioBaseDataset):
    def __init__(
        self,
        student_information: List[Tuple],
        data_root: str,
        feature_save_dir: str,
        load_from_feature=True,
        sr=22050,
        block_size=4096,
        hop_size=2048,
    ) -> None:

        self.feature_save_dir = feature_save_dir
        self.load_from_feature = load_from_feature
        self.sr = sr
        self.block_size = block_size
        self.hop_size = hop_size

        super().__init__(student_information=student_information, data_root=data_root)

    def _load_data_path(self):

        for (sid, year, band) in self.student_information:

            feature_path = os.path.join(
                self.feature_save_dir, "{}_{}_{}.npy".format(year, band, sid)
            )
            audio_path = os.path.join(
                self.data_root, str(year), band, "{}/{}.mp3".format(sid, sid)
            )

            self.data_path[str(sid)] = (audio_path, feature_path)

    def _read_from_feature(self, feature_path, start, end):
        with open(feature_path, "rb") as f:
            feature = np.load(f)

        if start is None:
            start = 0
        else:
            start = np.round(start * self.sr / self.hop_size).astype(int)

        if end is not None:
            end = np.round(end * self.sr / self.hop_size).astype(int) + 1

        return feature[start:end]

    def _read_from_audio(self, data_path, start, end):
        # TODO: support flexible feature extraction
        audio_path, feature_path = data_path

        y, _ = librosa.load(audio_path, sr=self.sr)
        feature = extract_segment_feature(
            y, sr=self.sr, block_size=self.block_size, hop_size=self.hop_size
        )
        with open(feature_path, "wb") as f:
            np.save(f, feature)

        return self._read_from_feature(self, feature_path, start, end)
