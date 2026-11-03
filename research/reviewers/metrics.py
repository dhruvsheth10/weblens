"""In-sample detection metrics.

AUROC and partial AUROC come from research.reddit_adaptation.common.summarize.
TPR at a false-positive cap uses a threshold chosen on the same rows, so those
operating points are optimistic.
"""
from __future__ import annotations

import math

import numpy as np

from research.reddit_adaptation.common import summarize


def threshold_at_fpr(logits: np.ndarray, labels_ai: np.ndarray, cap: float) -> tuple[float, float, float]:
    """Return (threshold, tpr, achieved_fpr) for score >= threshold.

    The threshold is the most permissive cut whose empirical FPR is still <= cap.
    Ties are kept together. Selected on these same rows.
    """
    if not 0 <= cap < 1:
        raise ValueError('FPR cap must be in [0, 1)')
    logits = np.asarray(logits, dtype=float)
    labels_ai = np.asarray(labels_ai, dtype=bool)
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite logit')
    negatives = int((~labels_ai).sum())
    positives = int(labels_ai.sum())
    if negatives == 0 or positives == 0:
        raise ValueError('Need both classes')
    allowed = int(math.floor(negatives * cap))
    order = np.argsort(-logits, kind='mergesort')
    ranked_logits = logits[order]
    ranked_labels = labels_ai[order]
    false_positives = 0
    true_positives = 0
    best_tpr = 0.0
    best_threshold = float(np.nextafter(ranked_logits[0], np.inf))
    best_fpr = 0.0
    index = 0
    while index < len(ranked_logits):
        end = index + 1
        while end < len(ranked_logits) and ranked_logits[end] == ranked_logits[index]:
            end += 1
        false_positives += int((~ranked_labels[index:end]).sum())
        true_positives += int(ranked_labels[index:end].sum())
        if false_positives > allowed:
            break
        tpr = true_positives / positives
        if tpr >= best_tpr:
            best_tpr = float(tpr)
            best_threshold = float(ranked_logits[index])
            best_fpr = false_positives / negatives
        index = end
    return best_threshold, best_tpr, float(best_fpr)


def evaluate(logits, rows: list[dict]) -> dict:
    logits = np.asarray(logits, dtype=float)
    labels = np.array([row['label'] == 'ai' for row in rows])
    at_zero = summarize(logits, rows, 0.0)
    points = {}
    for cap in (0.01, 0.05):
        threshold, tpr, achieved = threshold_at_fpr(logits, labels, cap)
        scoped = summarize(logits, rows, threshold)
        points[f'{cap:.2f}'] = {
            'threshold': threshold,
            'tpr': tpr,
            'achievedFpr': achieved,
            'selection': 'optimistic/in-sample: threshold chosen on this same set',
            'perCategoryFpr': {name: value['fpr'] for name, value in scoped['categories'].items()},
            'perCategory': scoped['categories'],
            'summarize': {key: scoped[key] for key in ('n', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'fpr')},
        }
    return {
        'n': len(rows),
        'positives': int(labels.sum()),
        'negatives': int((~labels).sum()),
        'auroc': at_zero['auroc'],
        'partialAuRoc01': at_zero['partialAuRoc01'],
        'partialAuRoc01Definition': 'sklearn.roc_auc_score max_fpr=0.01 via summarize (McClish-standardized partial AUC)',
        'averagePrecision': at_zero['averagePrecision'],
        'summarizeAtLogit0': at_zero,
        'optimisticInSample': points,
    }


def agreement(candidate, mirror, rows: list[dict]) -> dict:
    candidate = np.asarray(candidate, dtype=float)
    mirror = np.asarray(mirror, dtype=float)
    labels = np.array([row['label'] == 'ai' for row in rows])
    candidate_threshold, _, _ = threshold_at_fpr(candidate, labels, 0.01)
    mirror_threshold, _, _ = threshold_at_fpr(mirror, labels, 0.01)
    candidate_flag = candidate >= candidate_threshold
    mirror_flag = mirror >= mirror_threshold
    disagree = candidate_flag != mirror_flag
    candidate_right = int((disagree & (candidate_flag == labels)).sum())
    mirror_right = int((disagree & (mirror_flag == labels)).sum())
    return {
        'n': len(rows),
        'candidateThreshold': candidate_threshold,
        'mirrorThreshold': mirror_threshold,
        'selection': 'each model uses its own optimistic/in-sample 1% FPR threshold on this set',
        'disagreements': int(disagree.sum()),
        'candidateRight': candidate_right,
        'mirrorRight': mirror_right,
    }
