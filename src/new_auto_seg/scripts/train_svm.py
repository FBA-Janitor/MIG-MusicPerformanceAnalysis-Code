import pickle
import warnings

import numpy as np
from sklearn.svm import SVC

from utils.default_configs_path import (
    model_save_path,
    feature_write_dir,
    train_csv_list
)
from utils.utils import *

warnings.filterwarnings("ignore")


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
    X_train, y_train = load_data(train_csv_list, feature_dir, sr, block_size, hop_size)
    X_train = normalize_data(X_train)
    print("Training: {} frames!".format(len(y_train)))
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print("Training completed! Model saved at {}!".format(model_save_path))
        

if __name__ == '__main__':
    import fire

    fire.Fire()