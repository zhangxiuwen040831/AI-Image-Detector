from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.0
    return {"accuracy": acc, "auc": auc}
