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

from .scoreCommon import ScoreException, loadSegmentationImage, assertBinaryImage, computeTFPN


def matchInputFile(truthFile: pathlib.Path, predictionPath: pathlib.Path) -> pathlib.Path:
    # truthFile ~= 'ISIC_0000003_Segmentation.png'
    truthFileId = truthFile.stem.split('_')[1]

    predictionFileCandidates = [
        predictionFile
        for predictionFile in predictionPath.iterdir()
        if truthFileId in predictionFile.stem
    ]

    if not predictionFileCandidates:
        raise ScoreException(f'No matching submission for: {truthFile.name}')
    elif len(predictionFileCandidates) > 1:
        raise ScoreException(f'Multiple matching submissions for: {truthFile.name}')
    return predictionFileCandidates[0]


def scoreP1Image(truthFile: pathlib.Path, predictionFile: pathlib.Path) -> list:
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

    jaccard = truePositive / (truePositive + falseNegative + falsePositive)
    thresholdJaccard = jaccard if jaccard >= 0.65 else 0.0

    metrics = [
        {
            'name': 'threshold_jaccard',
            'value': thresholdJaccard
        }
    ]
    return metrics


def scoreP1(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> list:
    # Iterate over each file and call scoring executable on the pair
    scores = []
    for truthFile in sorted(truthPath.iterdir()):
        if truthFile.name in {'ATTRIBUTION.txt', 'LICENSE.txt'}:
            continue

        predictionFile = matchInputFile(truthFile, predictionPath)

        # truthFile ~= 'ISIC_0000003_segmentation.png'
        imageName = truthFile.stem.rsplit('_', 1)[0]
        metrics = scoreP1Image(truthFile, predictionFile)

        # TODO: maybe only return an 'aggregate' metric, rather than per-image scores
        scores.append({
            'dataset': imageName,
            'metrics': metrics
        })

    return scores
