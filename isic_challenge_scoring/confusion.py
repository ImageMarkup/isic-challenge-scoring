# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def createBinaryConfusionMatrix(
        truthBinaryValues: np.ndarray, predictionBinaryValues: np.ndarray, name=None) -> pd.Series:
    # This implementation is:
    # ~30x faster than sklearn.metrics.confusion_matrix
    # ~25x faster than sklearn.metrics.confusion_matrix(labels=[False, True])
    # ~6x faster than pandas.crosstab
    truthBinaryValues = truthBinaryValues.ravel()
    predictionBinaryValues = predictionBinaryValues.ravel()

    truthBinaryNegativeValues = 1 - truthBinaryValues
    testBinaryNegativeValues = 1 - predictionBinaryValues

    truePositive = np.sum(np.logical_and(truthBinaryValues,
                                         predictionBinaryValues))
    trueNegative = np.sum(np.logical_and(truthBinaryNegativeValues,
                                         testBinaryNegativeValues))
    falsePositive = np.sum(np.logical_and(truthBinaryNegativeValues,
                                          predictionBinaryValues))
    falseNegative = np.sum(np.logical_and(truthBinaryValues,
                                          testBinaryNegativeValues))

    # Storing the matrix as a Series instead of a DataFrame makes it easier to reference elements
    # and aggregate multiple matrices
    cm = pd.Series({
        'TP': truePositive,
        'TN': trueNegative,
        'FP': falsePositive,
        'FN': falseNegative,
    }, name=name)

    return cm


def normalizeConfusionMatrix(cm: pd.Series) -> pd.Series:
    return cm / cm.sum()
