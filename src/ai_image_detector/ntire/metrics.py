from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def _to_numpy(x: Iterable) -> np.ndarray:
    return np.asarray(list(x), dtype=float)


def expected_calibration_error(y_true: Iterable[int], y_prob: Iterable[float], bins: int = 15) -> float:
    y_t = _to_numpy(y_true).astype(int)
    y_p = _to_numpy(y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_t)
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (y_p >= left) & (y_p < right if i < bins - 1 else y_p <= right)
        if not np.any(mask):
            continue
        acc = y_t[mask].mean()
        conf = y_p[mask].mean()
        ece += (mask.sum() / max(n, 1)) * abs(acc - conf)
    return float(ece)


def compute_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> Dict[str, float]:
    y_t = _to_numpy(y_true).astype(int)
    y_p = _to_numpy(y_prob)
    y_hat = (y_p >= threshold).astype(int)
    metrics: Dict[str, float] = {}
    metrics["f1"] = float(f1_score(y_t, y_hat, zero_division=0))
    metrics["precision"] = float(precision_score(y_t, y_hat, zero_division=0))
    metrics["recall"] = float(recall_score(y_t, y_hat, zero_division=0))
    metrics["auroc"] = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else float("nan")
    metrics["auprc"] = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else float("nan")
    metrics["ece"] = expected_calibration_error(y_t, y_p, bins=15)
    return metrics
