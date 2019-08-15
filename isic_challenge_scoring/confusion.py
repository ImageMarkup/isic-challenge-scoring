from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def create_binary_confusion_matrix(
    truth_binary_values: np.ndarray,
    prediction_binary_values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    name: Optional[Union[str, Tuple[str, ...]]] = None,
) -> pd.Series:
    # This implementation is:
    # ~30x faster than sklearn.metrics.confusion_matrix
    # ~25x faster than sklearn.metrics.confusion_matrix(labels=[False, True])
    # ~6x faster than pandas.crosstab
    truth_binary_values = truth_binary_values.ravel()
    prediction_binary_values = prediction_binary_values.ravel()
    if weights is not None:
        weights = weights.ravel()

    truth_binary_negative_values = 1 - truth_binary_values
    test_binary_negative_values = 1 - prediction_binary_values

    true_positives = np.logical_and(truth_binary_values, prediction_binary_values)
    true_negatives = np.logical_and(truth_binary_negative_values, test_binary_negative_values)
    false_positives = np.logical_and(truth_binary_negative_values, prediction_binary_values)
    false_negatives = np.logical_and(truth_binary_values, test_binary_negative_values)

    if weights is not None:
        true_positives = true_positives * weights
        true_negatives = true_negatives * weights
        false_positives = false_positives * weights
        false_negatives = false_negatives * weights

    true_positive = np.sum(true_positives)
    true_negative = np.sum(true_negatives)
    false_positive = np.sum(false_positives)
    false_negative = np.sum(false_negatives)

    # Storing the matrix as a Series instead of a DataFrame makes it easier to reference elements
    # and aggregate multiple matrices
    cm = pd.Series(
        {'TP': true_positive, 'TN': true_negative, 'FP': false_positive, 'FN': false_negative},
        name=name,
    )

    return cm


def normalize_confusion_matrix(cm: pd.Series) -> pd.Series:
    return cm / cm.sum()
