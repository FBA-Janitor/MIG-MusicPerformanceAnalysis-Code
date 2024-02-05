"""
Feb 04, 2024 @suncerock

This file is used to train an SVM model that do frame-wise classification
"""

import pickle
import warnings

import numpy as np
from sklearn.svm import SVC

from utils.default_configs_path import (
    model_save_path,
    feature_write_dir,
    train_csv_list
)
from utils.utils import _load_data, _normalize_data

warnings.filterwarnings("ignore")


def train_svm(
    train_csv_list=train_csv_list,
    feature_dir=feature_write_dir,
    model_save_path=model_save_path,

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
    model_save_path : str, optional
        path to save the model, by default model_save_path
    sr : int, optional
        sampling rate, only used if no extracted features found, by default 22050
    block_size : int, optional
        block size for computing stft, only used if no extracted features found, by default 4096
    hop_size : int, optional
        hop size for computing stft, only used if no extracted features found, by default 2048

    Return
    ----------
    None
    """
    X_train, y_train = _load_data(train_csv_list, feature_dir, sr, block_size, hop_size)
    X_train = _normalize_data(X_train, force_regenerate=True)
    print("Training: {} frames!".format(len(y_train)))
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print("Training completed! Model saved at {}!".format(model_save_path))
        

if __name__ == '__main__':
    import fire

    fire.Fire()