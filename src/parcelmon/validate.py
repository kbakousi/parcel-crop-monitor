"""Accuracy assessment of crop-group predictions.

Reports the confusion matrix, overall accuracy and per-class precision,
recall and F1 (Olofsson et al. 2014). When the reference is the farmers'
declaration, these figures measure agreement with declarations, which
contain errors; they are not accuracy against field truth.

References
----------
Olofsson, P., Foody, G.M., Herold, M., Stehman, S.V., Woodcock, C.E., Wulder,
    M.A. (2014). Good practices for estimating area and assessing accuracy
    of land change. Remote Sensing of Environment 148, 42-57.
    https://doi.org/10.1016/j.rse.2014.02.015
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

__all__ = ["accuracy_report"]


def accuracy_report(y_true, y_pred, labels=None) -> dict:
    """Return ``overall_accuracy``, ``macro_f1``, ``per_class`` and ``confusion``.

    ``confusion`` rows are the reference, columns the prediction.
    """
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred)) if labels is None else [str(x) for x in labels]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    cm.index.name, cm.columns.name = "reference", "predicted"
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion": cm,
    }
