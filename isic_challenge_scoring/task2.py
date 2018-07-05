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

from .scoreCommon import ScoreException, assertBinaryImage, loadSegmentationImage, computeTFPN


def matchInputFile(truthFile: pathlib.Path, predictionPath: pathlib.Path) -> pathlib.Path:
    # truthFile ~= 'ISIC_0000003_attribute_streaks.png
    truthFileId = truthFile.stem.split('_')[1]
    truthFileAttribute = truthFile.stem.split('_')[3]

    predictionFileCandidates = [
        predictionFile
        for predictionFile in predictionPath.iterdir()
        if truthFileId in predictionFile.stem and truthFileAttribute in predictionFile.stem
    ]

    if not predictionFileCandidates:
        raise ScoreException(f'No matching submission for: {truthFile.name}')
    elif len(predictionFileCandidates) > 1:
        raise ScoreException(f'Multiple matching submissions for: {truthFile.name}')
    return predictionFileCandidates[0]


def scoreP2(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> list:
    truePositiveTotal = 0.0
    trueNegativeTotal = 0.0
    falsePositiveTotal = 0.0
    falseNegativeTotal = 0.0

    for truthFile in sorted(truthPath.iterdir()):
        if truthFile.name in {'ATTRIBUTION.txt', 'LICENSE.txt'}:
            continue

        predictionFile = matchInputFile(truthFile, predictionPath)

        truthImage = loadSegmentationImage(truthFile)
        truthImage = assertBinaryImage(truthImage, truthFile.name)

        predictionImage = loadSegmentationImage(predictionFile)
        if predictionImage.shape[0:2] != truthImage.shape[0:2]:
            raise ScoreException(
                f'Image {predictionFile.name} has dimensions {predictionImage.shape[0:2]}; '
                f'expected {truthImage.shape[0:2]}.')

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

    jaccardTotal = truePositiveTotal / (truePositiveTotal + falseNegativeTotal + falsePositiveTotal)
    scores = [
        {
            'dataset': 'aggregate',
            'metrics': [
                {
                    'name': 'jaccard',
                    'value': jaccardTotal
                }
            ]
        }
    ]
    return scores
