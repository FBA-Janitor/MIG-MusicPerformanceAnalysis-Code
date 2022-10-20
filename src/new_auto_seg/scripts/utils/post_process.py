import numpy as np

SMOOTH_LABEL = 'old'           # slower using old code

def smooth_label(pred):
    """
    Smooth the prediction label to remove short segments of performance or silence
    FIXME: use a graceful way to smooth and debug ISSUE#7
    TODO: add arguments, min_perform_frame and min_silence_frame
    Input:
        pred: np.ndarray, (num_frame,)
    Output:
        pred: same shape as input, smoothed
    """
    if SMOOTH_LABEL == 'old':
        pred[0] = 0
        pred[-1] = 0

        pred_diff = np.diff(pred)
        pred_fixed = np.copy(pred)

        m3 = np.where(pred_diff==1)[0]
        if np.sum(m3 >= len(pred_diff) - 2) > 0:
            m3 = m3[:-np.sum(m3 >= len(pred_diff) - 1)]
        m4 = m3[np.where(pred_diff[m3 + 1] == -1)[0]] + 1
        pred_fixed[m4] = 0.

        m1 = np.where(pred_diff == -1)[0]
        if np.sum(m1 >= len(pred_diff) - 2) > 0:
            m1 = m1[:-np.sum(m1 >= len(pred_diff) - 1)]
        m2 = m1[np.where(pred_diff[m1 + 1] == 1)[0]] + 1
        pred_fixed[m2] = 1.0

        pred = pred_fixed  # smoothed label
    else:
        raise NotImplementedError("New label smoothing note implemented yet!")
    return pred

def pred2seg(pred, time_stamp):
    """
    Convert the raw prediction output to the time_stamp
    Input:
        pred: np.ndarray, (num_frame, ), prediction of model
        time_stamp: np.ndarray, (num_frame, ), corresponding time stamp
    Output:
        seg: (num_seg, 2), start time and duration of each segments
    """
    # TODO: check how the time_stamp match
    begin_diff = np.diff(pred, append=0)
    # begin_diff = np.diff(pred, prepend=0)
    begin_time = time_stamp[begin_diff == 1]
    end_diff = np.diff(pred)
    end_time = time_stamp[1:][end_diff == -1]
    assert len(begin_time) == len(end_time)

    return np.vstack([begin_time, end_time - begin_time]).T

def combine_seg(
    seg,
    target_count,
    min_mus_time=3.0
    ):

    mus_begin, mus_duration = seg[:, 0], seg[:, 1]
    mus_end = mus_begin + mus_duration

    # remove the music segments < min_mus_time
    if np.sum(mus_duration > min_mus_time) >= target_count:
        seg = seg[mus_duration > min_mus_time]
    else:
        seg_keep = np.argsort(mus_duration)[-target_count:]
        seg = seg[np.sort(seg_keep)]
        return seg
    
    # remove the shortest non_music segments
    non_mus_end = seg[1:, 0]
    non_mus_begin = seg[:-1, 0] + seg[:-1, 1]
    non_mus_duration = non_mus_end - non_mus_begin

    seg_keep = np.argsort(non_mus_duration)[-(target_count - 1):]
    non_mus_end = np.sort(non_mus_end[seg_keep])
    non_mus_begin = np.sort(non_mus_begin[seg_keep])

    mus_begin = np.insert(non_mus_end, 0, seg[0, 0])
    mus_end = np.append(non_mus_begin, seg[-1, 0] + seg[-1, 1])
    return np.vstack([mus_begin, mus_end - mus_begin, mus_end]).T
