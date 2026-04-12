"""Classification metric computation."""

from typing import Any, NamedTuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class ClassificationReport(NamedTuple):
    """Structured container for standard classification metrics."""

    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float


def evaluate_predictions(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
) -> ClassificationReport:
    """Compute a comprehensive set of classification metrics."""
    return ClassificationReport(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        precision_weighted=precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        recall_weighted=recall_score(y_true, y_pred, average="weighted", zero_division=0),
    )
