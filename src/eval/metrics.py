from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def _to_numpy(y: Iterable) -> np.ndarray:
    arr = np.asarray(list(y), dtype=float)
    return arr


def compute_binary_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
) -> Dict[str, float]:
    """
    计算二分类常用指标：
    - accuracy
    - f1
    - auroc
    - auprc
    """
    y_true_arr = _to_numpy(y_true).astype(int)
    y_prob_arr = _to_numpy(y_prob)
    y_pred_arr = (y_prob_arr >= 0.5).astype(int)

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
    metrics["f1"] = float(f1_score(y_true_arr, y_pred_arr))

    try:
        metrics["auroc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = float(
            average_precision_score(y_true_arr, y_prob_arr)
        )
    except ValueError:
        metrics["auprc"] = float("nan")

    return metrics


def summarize_metrics(
    results: Dict[str, Dict[str, float]]
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    对多个数据集上的指标取平均（用于跨模型平均准确率等）。

    返回：
    - mean_metrics: 各指标的平均值
    - per_dataset: 每个数据集的详细指标
    """
    if not results:
        return {}, {}

    keys = list(next(iter(results.values())).keys())
    mean_metrics = {
        k: float(np.nanmean([results[d][k] for d in results])) for k in keys
    }
    return mean_metrics, results

