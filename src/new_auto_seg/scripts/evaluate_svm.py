import pickle
import os
from typing import List
import warnings
from tqdm import tqdm

import numpy as np

from utils.default_configs_path import (
    model_load_path,
    feature_write_dir,
    evaluate_csv_list
)
from utils.utils import(
    _read_multiple_csv,
    _read_annotation_as_seg,
    _load_data,
    _normalize_data
)


# TODO: add deviation of start/end points

def _calculate_acc(y_pred, y_eval, macro=True):
    """
    Compute the accuracy of the model
    If macro = False, then calculate the frame-wise accuracy
    If macro = True, then the accuracy of each class is first calculated,
    then compute the average of classes
    This is an internal function that should not be called by users.

    Parameters
    ----------
    y_pred : np.adarray
        (len_seg, ), binary output of the model
    y_eval : np.ndarray
        (len_seg, ), binary ground truth

    Return
    ---------
    float
        the accuracy of model

    """
    if not macro:
        return (y_pred == y_eval).mean()
    else:
        pos_acc = (y_pred[y_eval == 1] == y_eval[y_eval == 1]).mean()
        neg_acc = (y_pred[y_eval == 0] == y_eval[y_eval == 0]).mean()
        return (pos_acc + neg_acc) / 2

def _calculate_PRF(y_pred, y_eval):
    """
    Compute the precision, recall and the F1 score of the model.
    This is an internal function that should not be called by users.

    Parameters
    ----------
    y_pred : np.adarray
        (len_seg, ), binary output of the model
    y_eval : np.ndarray
        (len_seg, ), binary ground truth

    Returns
    ---------
    (float, float, float)
        precision, recall and F1
    """

    precision = np.count_nonzero((y_pred == 1) & (y_eval == 1)) / np.count_nonzero(y_pred == 1)
    recall = np.count_nonzero((y_pred == 1) & (y_eval == 1)) / np.count_nonzero(y_eval == 1)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
   
def _calculate_seg_IoU(seg_output, seg_target):
    output_start, output_end = seg_output
    target_start, target_end = seg_target

    intersec_start = max(output_start, target_start)
    intersec_end = min(output_end, target_end)

    if intersec_end <= intersec_start:
        return 0

    union_start = min(output_start, target_start)
    union_end = max(output_end, target_end)

    return (intersec_end - intersec_start) / (union_end - union_start)
 
def _calculate_piece_IoU(output, target):
    max_length = max(output[-1][1], target[-1][1])
    time_frame = np.arange(0, max_length, 0.1)
    output_frame = np.zeros_like(time_frame)
    target_frame = np.zeros_like(time_frame)

    for output_start, output_end in output:
        output_frame[(time_frame > output_start) & (time_frame < output_end)] = 1
    for target_start, target_end in target:
        target_frame[(time_frame > target_start) & (time_frame < target_end)] = 1

    intersec = np.count_nonzero(output_frame * target_frame)
    union = np.count_nonzero(output_frame + target_frame)
    return intersec / union


def evaluate_svm(
    evaluate_csv_list=evaluate_csv_list,
    feature_dir=feature_write_dir,

    model_path=model_load_path,
    sr=22050,
    block_size=4096,
    hop_size=2048,

    requires_accuracy=True,
    requires_prf=True
):
    """
    Evaluate the accuracy of the SVM model (instead of the segmentation system)

    Parameters
    ----------
    evaluate_csv_list : str | List[str]
        the audio-segment pair metadata csv file or a list of csv files. Default: evaluate_csv_list (in the default config file)
    feature_dir : str
        path to save the feature data
    model_path : str
        path of the saved model
    sr : int
        sampling rate
    block_size : int
        block size when computing feature
    hop_size : int
        hop size when computing feature
    requires_accuracy : bool
        whether to compute accuracy
    requires_prf : bool
        whether to compute precision, recall and F1
    
    Returns
    ----------
    None
    """
    if isinstance(evaluate_csv_list, str):
        evaluate_csv_list = [evaluate_csv_list]

    X_eval, y_eval = _load_data(evaluate_csv_list, feature_dir, sr, block_size, hop_size)
    X_eval = _normalize_data(X_eval)
    print("Evaluating: {} frames!".format(len(y_eval)))
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_eval)

    if requires_accuracy:
        acc_mic = _calculate_acc(y_pred=y_pred, y_eval=y_eval, macro=False)
        acc_mac = _calculate_acc(y_pred=y_pred, y_eval=y_eval, macro=True)
        print("Frame-wise accuracy = {:.2f}%".format(acc_mic * 100))
        print("Class-wise accuracy = {:.2f}%".format(acc_mac * 100))
    
    if requires_prf:
        precision, recall, f1 = _calculate_PRF(y_pred=y_pred, y_eval=y_eval)
        print("Precision={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(precision, recall, f1))

def evaluate_seg(
    result_dir,
    evaluate_csv_list: str | List[str] = evaluate_csv_list,

    ignore_fail_seg=False
):
    """
    Evaluate the IoU of the auto segmentation system

    Parameters
    ----------
    result_dir : str
        path to the directory with segmentation results
    evaluate_csv_list : str | List[str]
        the audio-segment pair metadata csv file or a list of csv files. Default: evaluate_csv_list (in the default config file)
    
    Returns
    ----------
    None
    """
    if isinstance(evaluate_csv_list, str):
        evaluate_csv_list = [evaluate_csv_list]
        
    data_csv = _read_multiple_csv(evaluate_csv_list)
    res_seg_IoU, res_piece_IoU = [], []
    n_all, n_success = 0, 0
    for stu_id, _, seg_csv in tqdm(data_csv):
        res_csv = os.path.join(result_dir, "{}.csv".format(stu_id))
        if not os.path.exists(res_csv):
            warnings.warn("Results for student {} not found!".format(stu_id))
            continue
        
        output = _read_annotation_as_seg(res_csv)
        target = _read_annotation_as_seg(seg_csv)

        if len(output) != len(target) and ignore_fail_seg:
            continue

        n_all += 1
        piece_IoU = _calculate_piece_IoU(output, target)
        res_piece_IoU.append(piece_IoU)

        if len(output) != len(target):
            continue

        n_success += 1
        num_exercises = len(output)
        seg_IoU = [_calculate_seg_IoU(output[i], target[i]) for i in range(num_exercises)]
        res_seg_IoU.append(seg_IoU)

    res_seg_IoU = np.array(res_seg_IoU)
    res_piece_IoU = np.array(res_piece_IoU)

    print("Segment IoU", res_seg_IoU.mean(axis=0))
    print("Piece IoU", res_piece_IoU.mean())
    print("Success percentage: {}/{} = {:.2f}%".format(n_success, n_all, n_success / n_all * 100))
      

if __name__ == '__main__':
    import fire

    fire.Fire()