# coding=utf-8

import os

import numpy as np
from sklearn.metrics import average_precision_score


def matchInputFile(truthFile, testDir):
    # truthFile ~= 'ISIC_0000003_Segmentation.png' (p1)
    # truthFile ~= 'ISIC_0000003.json' (P2)
    truthFileId = os.path.splitext(truthFile)[0].split('_')[1]

    testPathCandidates = [
        os.path.join(testDir, testFile)
        for testFile in os.listdir(testDir)
        if truthFileId in testFile
    ]

    if not testPathCandidates:
        raise Exception('No matching submission for: %s' % truthFile)
    elif len(testPathCandidates) > 1:
        raise Exception('Multiple matching submissions for: %s' % truthFile)
    return testPathCandidates[0]


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
            'value': (float(truePositive + trueNegative) /
                      float(truePositive + trueNegative +
                           falsePositive + falseNegative))
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
            'value': (float(truePositive) /
                      float(truePositive + falseNegative + falsePositive))
        },
        {
            'name': 'dice',
            'value': (float(2 * truePositive) /
                      float(truthValuesSum + testValuesSum))
        }
    ]
    return metrics

def computeAveragePrecisionMetrics(truthValues, testValues):
    """
    Compute average precision.
    """
    metrics = [
        {
            'name': 'average_precision',
            'value': average_precision_score(
                y_true=truthValues, y_score=testValues)
        }
    ]
    return metrics
