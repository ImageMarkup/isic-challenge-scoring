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

from scoreCommon import ScoreException, matchInputFile, loadSegmentationImage, \
    assertBinaryImage, computeCommonMetrics, computeSimilarityMetrics


def scoreP1Image(truthPath, testPath):
    truthImage = loadSegmentationImage(truthPath)
    truthImage = assertBinaryImage(truthImage, os.path.basename(truthPath))
    testImage = loadSegmentationImage(testPath)

    if testImage.shape[0:2] != truthImage.shape[0:2]:
        raise ScoreException('Image %s has dimensions %s; expected %s.' %
                             (os.path.basename(testPath), testImage.shape[0:2],
                              truthImage.shape[0:2]))

    truthBinaryImage = (truthImage > 128)
    testBinaryImage = (testImage > 128)

    metrics = computeCommonMetrics(truthBinaryImage, testBinaryImage)
    metrics.extend(computeSimilarityMetrics(truthBinaryImage, testBinaryImage))
    return metrics


def scoreP1(truthDir, testDir):
    # Iterate over each file and call scoring executable on the pair
    scores = []
    for truthFile in sorted(os.listdir(truthDir)):
        testPath = matchInputFile(truthFile, testDir)
        truthPath = os.path.join(truthDir, truthFile)

        # truthFile ~= 'ISIC_0000003_segmentation.png'
        imageName = truthFile.rsplit('_', 1)[0]
        metrics = scoreP1Image(truthPath, testPath)

        scores.append({
            'dataset': imageName,
            'metrics': metrics
        })

    return scores
