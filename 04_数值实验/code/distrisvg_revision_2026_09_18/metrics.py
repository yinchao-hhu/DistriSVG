"""Pair-level metrics with explicit empty-denominator and tied-score handling."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def bh(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("BH input must be a finite vector in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1)
    return out


def acat(p, axis=0):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return np.clip(0.5 - np.arctan(np.mean(np.tan(np.pi * (0.5 - p)), axis=axis)) / np.pi, 1e-15, 1)


def pair_metrics(truth, p, q=None):
    truth = np.asarray(truth, dtype=int)
    p = np.asarray(p, dtype=float)
    if truth.shape != p.shape or not np.isin(truth, [0, 1]).all():
        raise ValueError("Truth and p-values must have the same binary-pair layout")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values are not silently converted to one")
    q = bh(p) if q is None else np.asarray(q, dtype=float)
    if q.shape != p.shape or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Invalid q-values")
    selected, active = q < 0.05, truth == 1
    tp, fp = int(np.sum(selected & active)), int(np.sum(selected & ~active))
    fn, tn = int(np.sum(~selected & active)), int(np.sum(~selected & ~active))
    ratio = lambda a, b: a / b if b else float("nan")
    score = -np.log10(np.maximum(p, 1e-300))
    return dict(n_pairs=len(p), n_active=int(active.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                selected_pairs=tp + fp, power=ratio(tp, tp + fn), precision=ratio(tp, tp + fp),
                fdp=fp / max(tp + fp, 1), specificity=ratio(tn, tn + fp),
                f1=ratio(2 * tp, 2 * tp + fp + fn),
                auprc=float(average_precision_score(truth, score)) if active.any() else float("nan"),
                auroc=float(roc_auc_score(truth, score)) if 0 < active.sum() < len(truth) else float("nan"))
