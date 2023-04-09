import sys

import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score, roc_curve


def confusion_matrix_curve(y, yhat):
    p = np.sum(y)
    n = y.shape[0] - p
    fpr, tpr, threshold = roc_curve(y, yhat)
    tns = []
    fps = []
    fns = []
    tps = []
    for i in range(len(fpr)):
        fp = fpr[i] * n
        tp = tpr[i] * p
        tn = n - fp
        fn = p - tp
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
    return np.array(tns), np.array(fps), np.array(fns), np.array(tps), threshold


def compute_mcc(tns, fps, fns, tps):
    a = []
    for i in range(len(tns)):
        top = tps[i] * tns[i] - fps[i] * fns[i]
        if top == 0:
            a.append(0)
        else:
            bottom = np.sqrt((tps[i] + fps[i]) * (tps[i] + fns[i]) * (tns[i] + fps[i]) * (tns[i] + fns[i]))
            a.append(top / bottom)
    return np.array(a)


def find_best_threshold(y, yhat, scoring='balanced_accuracy'):
    if scoring == 'balanced_accuracy':
        fpr, tpr, threshold = roc_curve(y, yhat)
        thr = threshold[np.argmax(tpr - fpr)]  # max. tpr - fpr => max. balanced accuracy
        y_pred = np.array([1 if i >= thr else 0 for i in yhat])
    elif scoring == 'matthews_corrcoef':
        tns, fps, fns, tps, threshold = confusion_matrix_curve(y, yhat)
        mccs = compute_mcc(tns, fps, fns, tps)
        thr = threshold[np.argmax(mccs)]
        y_pred = np.array([1 if i >= thr else 0 for i in yhat])
    else:
        print('Unknown scoring', file=sys.stderr)
        assert 0
    return thr, y_pred


def print_thresholds(y, yhat, verbose=True):
    for s in ['matthews_corrcoef', 'balanced_accuracy']:
        thr, y_pred = find_best_threshold(y, yhat, scoring=s)
        bal_acc, acc, mcc = balanced_accuracy_score(y, y_pred), accuracy_score(y, y_pred), matthews_corrcoef(y, y_pred)
        if (verbose):
            print('Search for the best threshold by maximizing "', s, '":', sep='')
            print('Threshold =', thr, sep='\t')
            print('Bal. Acc. =', bal_acc, sep='\t')
            print('Accuracy =', acc, sep='\t')
            print('MCC score =', mcc, '\n', sep='\t')
    return thr, mcc, y_pred


def print_scores(y, y_pred):
    bal_acc, acc, mcc = balanced_accuracy_score(y, y_pred), accuracy_score(y, y_pred), matthews_corrcoef(y, y_pred)
    print('Bal. Acc. =', round(bal_acc, 3), sep='\t')
    print('Accuracy =', round(acc, 3), sep='\t')
    print('MCC score =', round(mcc, 3), '\n', sep='\t')
    return acc, mcc
