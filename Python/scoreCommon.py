# coding=utf-8

import numpy as np


def _computeTFPN(truthBinaryValues, testBinaryValues):
    truthBinaryNegativeValues = 1 - truthBinaryValues
    testBinaryNegativeValues = 1 - testBinaryValues

    truePositive = np.sum(np.logical_and(truthBinaryValues,
                                         testBinaryValues))
    trueNegative = np.sum(np.logical_and(truthBinaryNegativeValues,
                                         testBinaryNegativeValues))
    falsePositive = np.sum(np.logical_and(truthBinaryNegativeValues,
                                          testBinaryValues))
    falseNegative = np.sum(np.logical_and(truthBinaryValues,
                                          testBinaryNegativeValues))

    return truePositive, trueNegative, falsePositive, falseNegative


def computeCommonMetrics(truthBinaryValues, testBinaryValues):
    """
    Computes accuracy, sensitivity, and specificity.
    """
    truePositive, trueNegative, falsePositive, falseNegative = _computeTFPN(
        truthBinaryValues, testBinaryValues
    )

    metrics = [
        {
            'name': 'accuracy',
            'value': float(truePositive + trueNegative) /
                     float(truePositive + trueNegative +
                           falsePositive + falseNegative)
        },
        {
            'name': 'sensitivity',
            'value': float(truePositive) /
                     float(truePositive + falseNegative)
        },
        {
            'name': 'specificity',
            'value': float(trueNegative) /
                     float(trueNegative + falsePositive)
        }
    ]
    return metrics


def computeSimilarityMetrics(truthBinaryValues, testBinaryValues):
    """
    Computes Jaccard index and Dice coefficient.
    """
    truePositive, trueNegative, falsePositive, falseNegative = _computeTFPN(
        truthBinaryValues, testBinaryValues
    )
    truthValuesSum = np.sum(truthBinaryValues, dtype=np.int)
    testValuesSum = np.sum(testBinaryValues, dtype=np.int)

    metrics = [
        {
            'name': 'jaccard',
            'value': float(truePositive) /
                     float(truePositive + falseNegative + falsePositive)
        },
        {
            'name': 'dice',
            'value': float(2 * truePositive) /
                     float(truthValuesSum + testValuesSum)
        }
    ]
    return metrics
