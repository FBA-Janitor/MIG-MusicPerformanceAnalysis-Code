import glob
import os
import pickle
import re
from typing import Optional
from tqdm import tqdm
import warnings

import librosa
import sklearn
import numpy as np
import scipy

from utils.post_process import (
    combine_seg,
    smooth_label,
    pred2seg
)
from utils.default_configs_path import (
    model_path,
    feature_write_dir,
)
from utils.feature import *
from utils.utils import write_txt

warnings.filterwarnings("ignore")


def segment_audio(
    model: sklearn.svm.SVC,
    audio_path: Optional[str] = None,
    feature_path: Optional[str] = None,

    sr=22050,
    block_size=4096,
    hop_size=2048,
    feature_write_dir=feature_write_dir,

    num_exercise=5,
    max_seg_tolerant=15
    ):
    """
    Segment from raw audio or extracted feature

    Parameters
    ----------
    model : sklearn.svm.SVC
        svm model
    audio : Optional[str], optional
        path to the audio file, by default None
    feature : Optional[str], optional
        path to the feature of the audio, by default None
    sr : int, optional
        only used when using raw audio, sampling rate when reading the audio, by default 22050
    block_size : int, optional
        only used when using raw audio, block size for STFT, by default 4096
    hop_size : int, optional
        only used when using raw audio, hop size for STFT, by default 2048
    feature_write_dir : str, optional
        only used when using raw audio, directory to save the feature file
    num_exercise : int, optional
        number of the exercises in the audio, by default 5
    max_seg_tolerant : int, optional
        the maximal number of segments after stage 1, more segments would be labeled as fail, by default 15
    
    Returns
    ----------
    int
        -1 if fail, 1 if succeed after smoothing, 2 if succeed by combining segmentations
    np.ndarray
        (n_seg, 3) containing start time, duration and end time
    """

    assert audio_path is not None or feature_path is not None  # use either audio or feature
    if feature_path is None:
        _, basename = os.path.split(audio_path)
        stu_id, _ = re.match(r"(\d{5})(\.mp3)", basename).groups()

        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        feature, time_stamp = write_feature(y, stu_id, feature_write_dir, sr, block_size, hop_size)
    else:
        with open(feature_path, 'rb') as f:
            feature_save = np.load(f)
            feature = feature_save['feature']
            time_stamp = feature_save['time_stamp']

    pred = model.predict(feature)
    pred -= 1 # conver 1/2 to 0/1

    # stage 1 post-processing
    pred = smooth_label(pred)
    seg = pred2seg(pred, time_stamp)

    if len(seg) == num_exercise:  # correct after stage 1
        return 1, seg
    
    elif len(seg) < num_exercise or len(seg) > max_seg_tolerant:
        return -1, seg
    
    else:
        seg = combine_seg(seg, target_count=num_exercise, min_mus_time=3.)
        return 2, seg

def segment_dir(
    input_dir,
    output_dir,
    model_path=model_path,

    from_feature=False,
    sr=22050,
    block_size=4096,
    hop_size=2048,
    feature_write_dir=feature_write_dir,

    num_exercise=5,
    max_seg_tolerant=15
):
    """
    Segment the audio file or feature file in a directory

    Parameters
    ----------
    input_dir : str
        path to the directory of the audio or feature files
        If it is audio file, the structure is directory/stu_id/stu_id.mp3
        If it is feature file, the structure is directory/stu_id.npz
    output_dir : str
        output directory to write the report and the segmentation results
    model_path : str, optional
        path to the trained svm model, by default model_path
    from_feature : bool, optional
        whether to use the extracted feature, by default False
    sr : int, optional
        only used when from_feature = False, sampling rate used to extract the feature, by default 22050
    block_size : int, optional
        only used when from_feature = False, block size used to extract the feature, by default 4096
    hop_size : int, optional
        only used when from_feature = False, hop size used to extract the feature, by default 2048
    feature_write_dir : str, optional
        only used when from_feature = False, directory to write the extracted features, by default feature_write_dir
    num_exercise : int, optional
        number of the exercises in the audio, by default 5
    max_seg_tolerant : int, optional
        the maximal number of segments after stage 1, more segments would be labeled as fail, by default 15
    
    Returns
    ----------
    None
    """

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if from_feature:  # directory/stu_id.mp3
        files = sorted(glob.glob(os.path.join(input_dir, '*.npz'), recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(input_dir, '**/*.mp3'), recursive=True))
    
    stage_1_success = []
    stage_2_success = []
    fail_list = []
    for file in tqdm(files):
        if from_feature:
            _, basename = os.path.split(file)
            stu_id, _ = re.match(r"(\d{5})(\.npz)", basename).groups()
            success, seg = segment_audio(model, feature_path=file,
                num_exercise=num_exercise, max_seg_tolerant=max_seg_tolerant)
        else:
            _, basename = os.path.split(file)
            stu_id, _ = re.match(r"(\d{5})(\.mp3)", basename).groups()
            success, seg = segment_audio(model, audio_path=file,
                sr=sr, block_size=block_size, hop_size=hop_size, feature_dir=feature_write_dir,
                num_exercise=num_exercise, max_seg_tolerant=max_seg_tolerant)
        
        write_txt(seg, "{}.txt".format(stu_id), output_dir)
        if success == 1:
            stage_1_success.append(stu_id)
        elif success == 2:
            stage_2_success.append(stu_id)
        else:
            fail_list.append(stu_id)

    f = open(os.path.join(output_dir, 'report.txt'), 'w')

    # Count the files
    f.write("Total file count: {}\n".format(len(files)))
    f.write("First stage success: {}\n".format(len(stage_1_success)))
    f.write("Second stage success: {}\n".format(len(stage_2_success)))
    f.write("Failed: {}\n".format(len(fail_list)))
    f.write('\n')

    # Write the list
    f.write("First stage success file list:\n")
    for stu_id in stage_1_success:
        f.write('\t{}\n'.format(stu_id))
    f.write("Second stage success file list:\n")
    for stu_id in stage_2_success:
        f.write('\t{}\n'.format(stu_id))
    f.write("Failed file list:\n")
    for stu_id in fail_list:
        f.write('\t{}\n'.format(stu_id))
    
    print("Successful classification! Results in {}".format(output_dir))


if __name__ == "__main__":
    import fire

    fire.Fire()
