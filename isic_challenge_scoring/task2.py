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

import pathlib
from typing import Dict, List

from isic_challenge_scoring.scoreCommon import computeTFPN, iterImagePairs


def score(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> List[Dict]:
    truePositiveTotal = 0.0
    trueNegativeTotal = 0.0
    falsePositiveTotal = 0.0
    falseNegativeTotal = 0.0

    for truthImage, predictionImage, truthFileId in iterImagePairs(truthPath, predictionPath):
        truthBinaryImage = (truthImage > 128)
        predictionBinaryImage = (predictionImage > 128)

        truePositive, trueNegative, falsePositive, falseNegative = computeTFPN(
            truthBinaryImage, predictionBinaryImage)

        # Normalize all values, since image sizes vary
        imageSize = truthImage.shape[0] * truthImage.shape[1]
        assert imageSize == sum([truePositive, trueNegative, falsePositive, falseNegative])

        # TODO: represent this as a confusion matrix, to simply computations
        truePositiveTotal += truePositive / imageSize
        trueNegativeTotal += trueNegative / imageSize
        falsePositiveTotal += falsePositive / imageSize
        falseNegativeTotal += falseNegative / imageSize

    scores = [
        {
            'dataset': 'aggregate',
            'metrics': [
                {
                    'name': 'jaccard',
                    'value': truePositiveTotal /
                            (truePositiveTotal + falsePositiveTotal + falseNegativeTotal)
                },
                {
                    'name': 'dice',
                    'value': (2 * truePositiveTotal) /
                             ((2 * truePositiveTotal) + falsePositiveTotal + falseNegativeTotal)
                }
            ]
        }
    ]
    return scores
