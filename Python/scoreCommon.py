#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import os

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


class ScoreException(Exception):
    pass


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
        raise ScoreException('No matching submission for: %s' % truthFile)
    elif len(testPathCandidates) > 1:
        raise ScoreException('Multiple matching submissions for: %s' % truthFile)
    return testPathCandidates[0]


def loadSegmentationImage(imagePath):
    try:
        image = Image.open(imagePath)
    except Exception as e:
        raise ScoreException('Could not decode image "%s" because: "%s"' %
                             (os.path.basename(imagePath), str(e)))

    if image.mode == '1':
        # NumPy crashes if a 1-bit (black and white) image is directly
        # coerced to an array
        image = image.convert('L')

    if image.mode != 'L':
        raise ScoreException('Image %s is not single-channel (grayscale).' %
                             os.path.basename(imagePath))

    return image


def resizeImage(image, width, height):
    """Resize image to (width, height) using nearest neighbor interpolation."""
    return image.resize((width, height), Image.NEAREST)


def convertToNumPyArray(image, imagePath):
    """Convert PIL Image to NumPy array."""
    image = np.array(image)

    imageValues = set(np.unique(image))
    if imageValues <= {0, 255}:
        # Expected values
        pass
    elif len(imageValues) <= 2:
        # Binary image with high value other than 255 can be corrected
        highValue = (imageValues - {0}).pop()
        image /= highValue
        image *= 255
        if set(np.unique(image)) > {0, 255}:
            raise ScoreException('Image %s contains values other than 0 and '
                                 '255.' % os.path.basename(imagePath))
    else:
        raise ScoreException('Image %s contains values other than 0 and 255.' %
                             os.path.basename(imagePath))

    return image


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
            'value': ((float(truePositive) /
                       float(truePositive + falseNegative))
                      # sensitivity can't be calculated if all are negative
                      if np.any(truthBinaryValues)
                      else None)
        },
        {
            'name': 'specificity',
            'value': ((float(trueNegative) /
                       float(trueNegative + falsePositive))
                      # specificity can't be calculated if all are positive
                      if not np.all(truthBinaryValues)
                      else None)

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
            'value': ((float(truePositive) /
                       float(truePositive + falseNegative + falsePositive))
                      if (truePositive + falseNegative + falsePositive) != 0
                      else None)
        },
        {
            'name': 'dice',
            'value': ((float(2 * truePositive) /
                       float(truthValuesSum + testValuesSum))
                      if (truthValuesSum + testValuesSum) != 0
                      else None)
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


def computeAUCMetrics(truthValues, testValues):
    """
    Compute AUC measure.
    """
    metrics = [
        {
            'name': 'area_under_roc',
            'value': roc_auc_score(
                y_true=truthValues, y_score=testValues)
        }
    ]
    return metrics


def computeSPECMetrics(truthValues, testValues, sensitivityThreshold):
    """
    Compute specificity at specified sensitivity.
    """
    # Use sklearn to grab the ROC curve
    falsePositiveRates, truePositiveRates, thresholds = roc_curve(
        y_true=truthValues, y_score=testValues)

    # Search for the point along the curve where sensitivityThreshold occurs.
    for position, truePositiveRate in enumerate(truePositiveRates):
        if truePositiveRate >= sensitivityThreshold:
            falsePositiveRate = falsePositiveRates[position]
            trueNegativeRate = 1.0 - falsePositiveRate
            break
    else:
        trueNegativeRate = 0.0

    # Report the value
    metrics = [
        {
            # Metric names may not contain periods, in order for Covalic to
            # store title / description mappings for them
            'name': 'spec_at_sens_%s' %
                ('%g' % sensitivityThreshold * 100).replace('.', '_'),
            'value': trueNegativeRate  # This is specificity
        }
    ]
    return metrics
