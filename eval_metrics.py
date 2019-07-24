"""
@Overview: There are errors for computing eer and acc, when it comes to l2 and cosine distance.
For l2 distacne: when the distance is less than the theshold, it should be true;
For cosine distance: when the distance is greater than the theshold, it's true.
"""
import pdb

import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate

def evaluate(distances, labels):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 30, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, distances,
        labels)

    thresholds = np.arange(0, 30, 0.001)
    val,  far = calculate_val(thresholds, distances,
        labels, 1e-3)

    return tpr, fpr, accuracy, val,  far

def evaluate_eer(distances, labels):
    # Calculate evaluation metrics
    max_threshold = np.max(distances)
    thresholds = np.arange(0, max_threshold, 0.001)
    tpr, fpr, best_accuracy = calculate_roc(thresholds, distances, labels)
    err, accuracy = calculate_eer(thresholds, distances, labels)
    # thresholds = np.arange(0, 30, 0.001)
    # val,  far = calculate_val(thresholds, distances,
    #     labels, 1e-3)
    return err, best_accuracy

def evaluate_kaldi_eer(distances, labels, cos=True):
    """
    The distance score should be larger when two samples are more similar.
    :param distances:
    :param labels:
    :param cos:
    :return:
    """
    # split the target and non-target distance array
    target = []
    non_target = []

    for (distance, label) in zip(distances, labels):
        if not cos:
            distance = -distance
        if label:
            target.append(distance)
        else:
            non_target.append(distance)

    target = np.sort(target)
    non_target = np.sort(non_target)

    target_size = target.size
    target_position = 0
    # pdb.set_trace()
    while(target_position+1 < target_size):
        nontarget_size = non_target.size
        nontarget_n = nontarget_size * target_position * 1.0 / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)

        if (nontarget_position < 0):
            nontarget_position = 0
        # The exceptions from non targets are samples where cosine score is > the target score
        if (non_target[nontarget_position] < target[target_position]):
            break

        target_position += 1

    threshold = target[target_position]
    eer = target_position * 1.0 / target_size

    max_threshold = np.max(distances)
    thresholds = np.arange(0, max_threshold, 0.001)
    tpr, fpr, best_accuracy = calculate_roc(thresholds, distances, labels)

    return eer, best_accuracy

def calculate_roc(thresholds, distances, labels):

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))
    acc_train = np.zeros((nrof_thresholds))
    accuracy = 0.0

    indices = np.arange(nrof_pairs)

    # Find the best threshold for the fold

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], acc_train[threshold_idx] = calculate_accuracy(threshold, distances, labels)
    best_threshold_index = np.argmax(acc_train)

    return tprs[best_threshold_index], fprs[best_threshold_index], acc_train[best_threshold_index]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    #fnr = 0 if (tp+fn==0) else float(fn) / float(tp+fn)

    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def calculate_eer(thresholds, distances, labels):

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))
    fnrs = np.zeros((nrof_thresholds))
    acc_train = np.zeros((nrof_thresholds))
    accuracy = 0.0

    indices = np.arange(nrof_pairs)

    # Find the threshold where fnr=fpr for the fold
    # Todo: And the highest accuracy??
    eer_index = 0
    fpr_fnr = 1.0
    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], fnrs[threshold_idx], acc_train[threshold_idx] = calculate_eer_accuracy(threshold, distances, labels)
        if np.abs(fprs[threshold_idx]-fnrs[threshold_idx])<fpr_fnr:
            eer_index = threshold_idx
            fpr_fnr = np.abs(fprs[threshold_idx]-fnrs[threshold_idx])

    #print("Threshold for the eer is {}.".format(thresholds[eer_index]))
    return  fnrs[eer_index], acc_train[eer_index]


def calculate_eer_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)

    fnr = 0 if (tp + fn == 0) else float(fn) / float(tp + fn)

    acc = float(tp + tn) / dist.size
    return tpr, fpr, fnr, acc

def calculate_val(thresholds, distances, labels, far_target=0.1):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)

    indices = np.arange(nrof_pairs)

    # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)

    for threshold_idx, threshold in enumerate(thresholds):
        _, far_train[threshold_idx] = calculate_val_far(threshold, distances, labels)
    if np.max(far_train)>=far_target:
        f = interpolate.interp1d(far_train, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = 0.0

    val, far = calculate_val_far(threshold, distances, labels)

    return val, far


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0,0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far