import csv
import pickle
import os
import warnings

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

def evaluate_svm_acc(
    evaluate_csv_list=evaluate_csv_list,
    feature_dir=feature_write_dir,

    model_path=model_load_path,
    sr=22050,
    block_size=4096,
    hop_size=2048,

    macro=False
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
    if not macro:
        print("Frame-wise accuracy = ", (y_pred == y_eval).mean())
    else:
        pos_acc = (y_pred[y_eval == 1] == y_eval[y_eval == 1]).mean()
        neg_acc = (y_pred[y_eval == 0] == y_eval[y_eval == 0]).mean()
        print("Class-wise accuracy = ", (pos_acc + neg_acc) / 2)


def calculate_PRF(output, target):
    output_start, output_end = output
    target_start, target_end = target

    overlap_start = max(output_start, target_start)
    overlap_end = min(output_end, target_end)
    
    precision = max(0, overlap_end - overlap_start) / (output_end - output_start)
    recall = max(0, overlap_end - overlap_start) / (target_end - target_start)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    
    return precision, recall, f1


def calculate_IoU(output, target):
    output_start, output_end = output
    target_start, target_end = target

    intersec_start = max(output_start, target_start)
    intersec_end = min(output_end, target_end)

    union_start = min(output_start, target_start)
    union_end = max(output_end, target_end)

    return (intersec_end - intersec_start) / (union_end - union_start)

def evaluate_segment_output(
    result_dir,
    evaluate_csv_list=evaluate_csv_list,
):
    data_csv = read_multiple_csv(evaluate_csv_list)
    metrics = []
    for stu_id, _, seg_csv in data_csv:
        res_csv = os.path.join(result_dir, "{}.csv".format(stu_id))
        if not os.path.exists(res_csv):
            warnings.warn("Results for student {} not found!".format(stu_id))
            continue
        
        output = read_annotation_as_seg(res_csv)
        target = read_annotation_as_seg(seg_csv)
        
        if not len(output) == len(target):
            warnings.warn("Wrong segment number for student {} not found!".format(stu_id))
            continue
        
        num_exercise = len(output)
        for i in range(num_exercise):
            try:
                metrics.append(calculate_PRF(output[i], target[i]))
            except:
                print(stu_id)
                raise Exception

    metrics = np.array(metrics)
    metrics = np.mean(metrics, axis=0)
    print("Precision={metrics[0]:.2f}\tRecall={metrics[1]:.2f}\tF1={metrics[2]:.2f}".format(metrics=metrics))
            

if __name__ == '__main__':
    import fire

    fire.Fire()