import csv
import os
import pickle
import warnings
from tqdm import tqdm

import librosa
import numpy as np
from sklearn.svm import SVC

from utils.default_configs_path import (
    model_save_path,
    feature_write_dir,
    train_csv_list
)
from utils.feature import *

warnings.filterwarnings("ignore")

def read_multiple_csv(csv_list):
    """
    Read multiple csv files into one list
    """
    mult_csv = []
    for csv_file in csv_list:
        with open(csv_file, 'r') as f:
            one_csv = csv.reader(f)
            for row in one_csv:
                mult_csv.append(row)
    return mult_csv

def read_annotation(time_stamp, seg_csv):
    """
    Read annotation file

    Parameters
    ----------
    time_stamp : np.ndarray
        the time stamp array corresponding to the segmentation file and the feature
    seg_csv : str
        path to the segmentation file

    Returns
    ----------
    np.ndarray
        a binary array containing the labels for training
    """
    seg = []
    with open(seg_csv, 'r') as f:
        seg_reader = csv.DictReader(f)
        for row in seg_reader:
            seg.append((row["Start"], row["End"]))
    y = np.zeros_like(time_stamp)
    for start, end in seg:
        y[(time_stamp > float(start)) & (time_stamp < float(end))] = 1
    return y

def load_train_data(
    train_csv_list,
    feature_dir,

    sr=22050,
    block_size=4096,
    hop_size=2048
):
    """
    Load the data for training

    Parameters
    ----------
    train_csv_list : List[str]
        list of path to the summary csv files for training
    feature_dir : str
        path to save the feature data, first check whether there is feature in it,
        write feature into this directory if no extracted features found
    sr : int, optional
        sampling rate, only used if no extracted features found, by default 22050
    block_size : int, optional
        block size for computing stft, only used if no extracted features found, by default 4096
    hop_size : int, optional
        hop size for computing stft, only used if no extracted features found, by default 2048

    Returns
    ----------
    np.ndarray
        (num_frame, num_feature), feature data
    np.ndarray
        (num_frame, ), label data
    """
    train_csv = read_multiple_csv(train_csv_list)
    X_train, y_train = [], []
    for stu_id, audio_path, segment_path in tqdm(train_csv):

        # Load feature data
        feature_path = os.path.join(feature_dir, "{}.npz".format(stu_id))
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                feature_save = np.load(f)
                feature = feature_save['feature']
                time_stamp = feature_save['time_stamp']
        else:
            # continue
            audio, _ = librosa.load(audio_path, sr=sr, mono=True)
            # write feature into feature_write_dir
            feature, time_stamp = write_feature(audio, stu_id, feature_dir, sr, block_size, hop_size)
        
        # Load segmentation annotation
        anno = read_annotation(time_stamp, segment_path)
        X_train.append(feature)
        y_train.append(anno)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    return X_train, y_train


def train_svm(
    train_csv_list=train_csv_list,
    feature_dir=feature_write_dir,

    sr=22050,
    block_size=4096,
    hop_size=2048
):
    """
    Load the data for training

    Parameters
    ----------
    train_csv_list : List[str], optional
        list of path to the summary csv files for training, by default train_csv_list
    feature_dir : str, optional
        path to save the feature data, first check whether there is feature in it,
        write feature into this directory if no extracted features found
        by default feature_write_dir
    sr : int, optional
        sampling rate, only used if no extracted features found, by default 22050
    block_size : int, optional
        block size for computing stft, only used if no extracted features found, by default 4096
    hop_size : int, optional
        hop size for computing stft, only used if no extracted features found, by default 2048

    Returns
    ----------
    np.ndarray
        (num_frame, num_feature), feature data
    np.ndarray
        (num_frame, ), label data
    """
    X_train, y_train = load_train_data(train_csv_list, feature_dir, sr, block_size, hop_size)
    
    print("Training: {} frames!".format(len(y_train)))
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    print("Training completed! Model saved at {}!".format(model_save_path))
    with open(model, "wb") as f:
        pickle.dump(model, f)
        

if __name__ == '__main__':
    import fire

    fire.Fire()