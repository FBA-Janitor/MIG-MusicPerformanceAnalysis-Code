import os
import warnings

import numpy as np
import librosa

import tqdm

from feature_extractor import FeatureExtractor

warnings.filterwarnings("ignore")  # librosa gives a warning when loading mp3
                                   # because of the default backend

# for 2013 - 2018 data whose file structure is
# year/group/xxxxx/xxxxx.mp3
# for 2018 -
# year/group/xxxxx.mp3

feature_extractor = FeatureExtractor(block_size=4096, hop_size=2048)

def write_feature_data(
    audio_dir: str,
    output_dir: str = None,
):
    """
    Args:
        - audio_dir: a directory of .mp3 files to annotate
        - output_dir: a directory to output the feature .npz files
    """
    if output_dir is None:
        output_dir = 'tmp'

    for stu_id in tqdm.tqdm(os.listdir(audio_dir)):
        # only work for data before 2018
        # audio_dir: year/group
        # if not filename.split('.')[-1] == 'mp3':
        #     warnings.warn("Not an mp3 file: {}".format(filename))
        #     continue
        audio_path = os.path.join(audio_dir, stu_id, "{}.mp3".format(stu_id))
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        feature = feature_extractor(y, sr)
        
        np.savez(os.path.join(output_dir, stu_id), feature=feature)


if __name__ == '__main__':
    audio_dir = './test_audio'
    output_dir = 'test_feature'
    write_feature_data(audio_dir, output_dir)
