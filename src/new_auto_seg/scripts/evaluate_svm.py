import pickle
import os
import warnings
from tqdm import tqdm

import numpy as np

from utils.default_configs_path import (
    model_load_path,
    feature_write_dir,
    evaluate_csv_list
)
from utils.utils import(
    read_multiple_csv,
    read_annotation_as_seg,
    load_data,
    normalize_data
)


# TODO: add deviation of start/end points

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
    Evaluate the accuracy of the SVM model

    Parameters
    ----------
    
    Returns
    ----------
    None
    """
    X_eval, y_eval = load_data(evaluate_csv_list, feature_dir, sr, block_size, hop_size)
    X_eval = normalize_data(X_eval)
    print("Evaluating: {} frames!".format(len(y_eval)))
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_eval)

    if requires_accuracy:
        acc_mic = calculate_acc(y_pred=y_pred, y_eval=y_eval, macro=False)
        acc_mac = calculate_acc(y_pred=y_pred, y_eval=y_eval, macro=True)
        print("Frame-wise accuracy = {:.2f}%".format(acc_mic * 100))
        print("Class-wise accuracy = {:.2f}%".format(acc_mac * 100))
    
    if requires_prf:
        precision, recall, f1 = calculate_PRF(y_pred=y_pred, y_eval=y_eval)
        print("Precision={:.4f}\tRecall={:.4f}\tF1={:.4f}".format(precision, recall, f1))

def calculate_acc(y_pred, y_eval, macro=True):
    """
    Compute the accuracy of the model
    If macro = False, then calculate the frame-wise accuracy
    If macro = True, then the accuracy of each class is first calculated,
    then compute the average of classes

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

def calculate_PRF(y_pred, y_eval):
    """
    Compute the precision, recall and the F1 score of the model

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

def evaluate_seg(
    result_dir,
    evaluate_csv_list=evaluate_csv_list,

    ignore_fail_seg=False
):
    """
    Evaluate the IoU of the auto segmentation system

    Parameters
    ----------
    
    Returns
    ----------
    None
    """
    data_csv = read_multiple_csv(evaluate_csv_list)
    res_seg_IoU, res_piece_IoU = [], []
    for stu_id, _, seg_csv in tqdm(data_csv):
        res_csv = os.path.join(result_dir, "{}.csv".format(stu_id))
        if not os.path.exists(res_csv):
            warnings.warn("Results for student {} not found!".format(stu_id))
            continue
        
        output = read_annotation_as_seg(res_csv)
        target = read_annotation_as_seg(seg_csv)

        if len(output) != len(target) and ignore_fail_seg:
            continue

        piece_IoU = calculate_piece_IoU(output, target)
        res_piece_IoU.append(piece_IoU)

        if len(output) != len(target):
            continue

        num_exercises = len(output)
        seg_IoU = [calculate_seg_IoU(output[i], target[i]) for i in range(num_exercises)]
        res_seg_IoU.append(seg_IoU)

    res_seg_IoU = np.array(res_seg_IoU)
    res_piece_IoU = np.array(res_piece_IoU)

    print("Segment IoU", res_seg_IoU.mean(axis=0))
    print("Piece IoU", res_piece_IoU.mean())
        
        
def calculate_seg_IoU(seg_output, seg_target):
    output_start, output_end = seg_output
    target_start, target_end = seg_target

    intersec_start = max(output_start, target_start)
    intersec_end = min(output_end, target_end)

    if intersec_end <= intersec_start:
        return 0

    union_start = min(output_start, target_start)
    union_end = max(output_end, target_end)

    return (intersec_end - intersec_start) / (union_end - union_start)
 
def calculate_piece_IoU(output, target):
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


if __name__ == '__main__':
    import fire

    fire.Fire()