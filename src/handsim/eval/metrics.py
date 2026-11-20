"""Classification metrics. Balanced accuracy is the headline number because
rest dominates NinaPro (~55 % of windows): a classifier that always says
"rest" would get ~55 % plain accuracy but only 1/n_classes balanced accuracy."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def summarize(y_true, y_pred, labels=None) -> dict:
    labels = np.unique(y_true) if labels is None else np.asarray(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "labels": labels.tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
