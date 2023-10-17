import numpy as np

"""
TODO: post-processing
- inform which short segments to remove
"""


def smooth_performance(
    pred,
    min_performance_frame=3,
):
    p_start, p_cur = None, 0
    while p_cur < len(pred):
        if pred[p_cur] == 1 and p_start is None:
            p_start = p_cur
        if pred[p_cur] == 0:
            if p_start is not None and p_cur - p_start < min_performance_frame:
                pred[p_cur:p_start] = 0
                print("Smoothing {}-frame performance".format(p_cur - p_start))
            p_start = None
        p_cur += 1
    return pred



def smooth_silence(
    pred,
    min_silence_frame=3,
):
    p_start, p_cur = None, 0

    while pred[p_cur] != 1: # skip the starting silence
        p_cur += 1

    while p_cur < len(pred):
        if pred[p_cur] == 0 and p_start is None:
            p_start = p_cur
        if pred[p_cur] == 1:
            if p_start is not None and p_cur - p_start < min_silence_frame:
                pred[p_cur:p_start] = 1
                print("Smoothing {}-frame silence".format(p_cur - p_start))
            p_start = None
        p_cur += 1
    return pred


def smooth_label(
    pred,
    min_performance_frame=3,
    min_silence_frame=3,
    smooth_performance_first=False,
):
    """
    Smooth the prediction label to remove short segments of performance or silence
    Now: smooth 1-0-1 and 0-1-0 segments
    TODO: use smart smoothing

    Parameters
    ----------
    pred : np.ndarray
        (num_frame, ), initial prediction, 0 or 1
    min_performance_frame : int, optional
        minimum length of performance segments, by default 3
    min_silence_frame : int, optional
        minimum length of non-performance segments, by default 3
    smooth_performance_first : bool, optional
        whether to remove the short performance segments then silence segments, by default True
    
    Return
    ----------
    pred : np.ndarray
        (num_frame, ), smoothed predictions
    """
    # Assume the first and the last frame is non-performance
    pred[0] = 0
    pred[-1] = 0

    # if smooth_performance_first:
    #     pred = smooth_performance(pred, min_performance_frame)
    #     pred = smooth_silence(pred, min_silence_frame)
    # else:
    #     pred = smooth_silence(pred, min_silence_frame)
    #     pred = smooth_performance(pred, min_performance_frame)
    for i in range(len(pred) - 2):
        if pred[i] == 0 and pred[i+1] == 1 and pred[i+2] == 0:
            pred[i+1] = 0
        elif pred[i] == 1 and pred[i+1] == 0 and pred[i+2] == 1:
            pred[i+1] = 1

    return pred


def pred2seg(pred, time_stamp):
    """
    Convert the raw prediction output to the time_stamp
    
    Parameters
    ----------
    pred : np.ndarray
        (num_frame, ), prediction of model
    time_stamp : np.ndarray
        (num_frame, ), corresponding time stamp of the starting time of each frame
    
    Return
    ----------
    seg: np.ndarray
        (num_seg, 3), start time, duration and end time of each segments
    """

    begin_diff = np.diff(pred, append=0)
    begin_time = time_stamp[begin_diff == 1]
    end_diff = np.diff(pred, prepend=0)
    end_time = time_stamp[end_diff == -1]

    assert len(begin_time) == len(end_time), pred

    return np.vstack([begin_time, end_time - begin_time, end_time]).T

def combine_seg(
    seg,
    target_count,
    min_performance_time=5.0
    ):
    """
    Remove short performance segments that are too short,
    then remove last shortest non-performance short.

    Parameters:
    ----------
    seg: np.ndarray
        (num_segments, 3), start, duration and end time for each segment
    target_count: int
        target num segments of the piece
    min_performance_time: float, optional
        filter out the performance segments that have duration < min_performance_time, by default 5.0
    """

    performance_begin, performance_duration, performance_end = seg[:, 0], seg[:, 1], seg[:, 2]

    # remove the music segments < min_mus_time
    if np.count_nonzero(performance_duration > min_performance_time) >= target_count:
        seg = seg[performance_duration > min_performance_time]
    else:
        seg_keep = np.argsort(performance_duration)[-target_count:]
        seg = seg[np.sort(seg_keep)]
        return seg
    
    # remove the shortest non_music segments
    non_performance_begin = seg[:-1, 2]
    non_performance_end = seg[1:, 0]
    non_performance_duration = non_performance_end - non_performance_begin

    seg_keep = np.argsort(non_performance_duration)[-(target_count - 1):]
    non_performance_end = np.sort(non_performance_end[seg_keep])
    non_performance_begin = np.sort(non_performance_begin[seg_keep])

    performance_begin = np.insert(non_performance_end, 0, seg[0, 0])
    performance_end = np.append(non_performance_begin, seg[-1, 2])
    return np.vstack([performance_begin, performance_end - performance_begin, performance_end]).T
