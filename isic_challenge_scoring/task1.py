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

import numpy as np

from isic_challenge_scoring import metrics
from isic_challenge_scoring.scoreCommon import computeTFPN, iterImagePairs


def scoreImage(truthBinaryImage: np.ndarray, predictionBinaryImage: np.ndarray) -> Dict:
    truthBinaryImage = truthBinaryImage.ravel()
    predictionBinaryImage = predictionBinaryImage.ravel()

    truePositive, trueNegative, falsePositive, falseNegative = computeTFPN(
        truthBinaryImage, predictionBinaryImage)

    jaccard = truePositive / (truePositive + falsePositive + falseNegative)
    thresholdJaccard = jaccard if jaccard >= 0.65 else 0.0

    return {
        'threshold_jaccard': thresholdJaccard,
        'jaccard': jaccard,
        'dice': (2 * truePositive) / ((2 * truePositive) + falsePositive + falseNegative),
        'sensitivity': metrics.binarySensitivity(truthBinaryImage, predictionBinaryImage),
        'specificity': metrics.binarySpecificity(truthBinaryImage, predictionBinaryImage),
        'accuracy': metrics.binaryAccuracy(truthBinaryImage, predictionBinaryImage),
    }


def score(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> List[Dict]:
    # Iterate over each file and call scoring executable on the pair
    scores = []

    for truthImage, predictionImage, truthFileId in iterImagePairs(truthPath, predictionPath):
        truthBinaryImage = (truthImage > 128)
        predictionBinaryImage = (predictionImage > 128)

        metrics = scoreImage(truthBinaryImage, predictionBinaryImage)

        # TODO: maybe only return an 'aggregate' metric, rather than per-image scores
        scores.append({
            'dataset': truthFileId,
            'metrics': [
                {
                    'name': metricName,
                    'value': metricValue
                }
                for metricName, metricValue in metrics.items()
            ]
        })

    return scores
