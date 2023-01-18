import glob
import os
import pickle
import re
from typing import Optional, Union
from tqdm import tqdm
import warnings

import librosa
import sklearn
import numpy as np

from utils.post_process import (
    combine_seg,
    smooth_label,
    pred2seg
)
from utils.default_configs_path import (
    model_load_path,
    feature_write_dir
)
from utils.feature import *
from utils.utils import normalize_data

warnings.filterwarnings("ignore")


def write_csv(
    seg,
    stu_id,
    output_dir
):
    """
    Write the segmentation results into a .csv file

    Parameters
    ----------
    seg : np.ndarray
        (num_segments, 3), the start, duration and end time for each segment
    stu_id : str
        student id
    output_dir : str
        directory to write the output csv

    Return
    ----------
    None
    """

    new_file = open(os.path.join(output_dir, "{}.csv".format(stu_id)), 'w')
    new_file.write("Start,Duration,End\n")
    for i in range(len(seg)):
        output = '{:.6f},{:.6f},{:.6f}\n'.format(seg[i][0], seg[i][1], seg[i][2])
        new_file.write(output)
    new_file.close()


def write_report(
    output_dir,
    files,
    stage_1_success,
    stage_2_success,
    fail_list
):
    """
    Generate the report of auto segmentation

    Parameters
    ----------
    output_dir : str
        path to the output directory
    files : List[str]
        all student id
    stage_1_success : List[str]
        student id of success after the first stage pre-processing
    stage_2_success : List[str]
        student id of success after the second stage pre-processing
    fail_list : List[str]
        student id of failing to auto segment

    Return
    ----------
    None

    """
    f = open(os.path.join(output_dir, 'report.csv'), 'w')
    f.write('StudentID,Status\n')

    stage_1_success = set(stage_1_success)
    stage_2_success = set(stage_2_success)
    fail_list = set(fail_list)
    
    for sid in sorted(list(stage_1_success | stage_2_success | fail_list)):
        if sid in stage_1_success:
            f.write('{},1\n'.format(sid))
        elif sid in stage_2_success:
            f.write('{},2\n'.format(sid))
        elif sid in fail_list:
            f.write('{},0\n'.format(sid))
    f.close()
    
    print("Successful classification! Results in {}".format(output_dir))


def segment_audio(
    model: Union[str, sklearn.svm.SVC, None] = model_load_path,
    audio_path: Optional[str] = None,
    feature_path: Optional[str] = None,

    sr=22050,
    block_size=4096,
    hop_size=2048,
    feature_write_dir=feature_write_dir,

    num_exercise=5,
    max_seg_tolerant=15,

    stage_2_post_process=True
):
    """
    Segment from raw audio or extracted feature

    Parameters
    ----------
    model : str or sklearn.svm.SVC
        (str) path to the svm model or
        (sklearn.svm.SVC) the svm model
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
    stage_2_post_process : bool, optional
        whether to do the second stage of post processing, by default True

    Returns
    ----------
    int
        -1 if fail, 1 if succeed after smoothing, 2 if succeed by combining segmentations
    np.ndarray
        (n_seg, 3) containing start time, duration and end time
    """

    assert audio_path is not None or feature_path is not None  # use either audio or feature
    if feature_path is None:    # use audio file
        _, basename = os.path.split(audio_path)
        stu_id, _ = re.match(r"(\d{5})(\.mp3)", basename).groups()

        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # write feature into feature_write_dir
        feature, time_stamp = write_feature(y, stu_id, feature_write_dir, sr, block_size, hop_size)
    else:   # use feature file
        with open(feature_path, 'rb') as f:
            feature_save = np.load(f)
            feature = feature_save['feature']
            time_stamp = feature_save['time_stamp']     

    if isinstance(model, str):
        with open(model, 'rb') as f:
            model = pickle.load(f) 
    assert isinstance(model, sklearn.svm.SVC),\
        "Expect model to be sklearn.svm.SVC, but got {}".format(type(model))

    feature = normalize_data(feature)
    pred = model.predict(feature)

    # stage 1 post-processing
    pred = smooth_label(pred)
    seg = pred2seg(pred, time_stamp)

    if not stage_2_post_process: # do not do second stage
        return 1, seg

    if len(seg) == num_exercise:  # correct after stage 1
        return 1, seg
    
    elif len(seg) < num_exercise or len(seg) > max_seg_tolerant:
        return -1, seg
    
    else:
        seg = combine_seg(seg, target_count=num_exercise, min_performance_time=5.)
        return 2, seg


def segment_dir(
    input_dir,
    output_dir,
    model_path=model_load_path,

    from_feature=False,
    sr=22050,
    block_size=4096,
    hop_size=2048,
    feature_write_dir=feature_write_dir,

    num_exercise=5,
    max_seg_tolerant=15,

    stage_2_post_process=True
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
    stage_2_post_process : bool, optional
        whether to do the second stage of post processing, by default True

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
                num_exercise=num_exercise, max_seg_tolerant=max_seg_tolerant, stage_2_post_process=stage_2_post_process)
        else:
            _, basename = os.path.split(file)
            stu_id, _ = re.match(r"(\d{5})(\.mp3)", basename).groups()
            success, seg = segment_audio(model, audio_path=file,
                sr=sr, block_size=block_size, hop_size=hop_size, feature_write_dir=feature_write_dir,
                num_exercise=num_exercise, max_seg_tolerant=max_seg_tolerant, stage_2_post_process=stage_2_post_process)
        
        write_csv(seg, stu_id, output_dir)
        if success == 1:
            stage_1_success.append(stu_id)
        elif success == 2:
            stage_2_success.append(stu_id)
        else:
            fail_list.append(stu_id)

    write_report(output_dir, files, stage_1_success, stage_2_success, fail_list)


if __name__ == "__main__":
    import fire

    fire.Fire()
