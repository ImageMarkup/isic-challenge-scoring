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

import csv
import os

import numpy as np

from scoreCommon import ScoreException, \
    computeCommonMetrics, computeAveragePrecisionMetrics, \
    computeAUCMetrics, computeSPECMetrics


def matchRowName(truthImageName, testValues):
    # truthImageName ~= 'ISIC_0000003'
    truthImageId = truthImageName.split('_')[1]

    testValueCandidates = [
        testValue
        for testValue in testValues
        if truthImageId in testValue['image']
    ]

    if not testValueCandidates:
        raise ScoreException('No matching submissions for: %s' % truthImageName)
    elif len(testValueCandidates) > 1:
        raise ScoreException('Multiple matching submissions for: %s' %
                        truthImageName)
    return testValueCandidates[0]


def scoreP3(truthDir, testDir):
    truthFile = os.path.join(truthDir, 'ISIC-2017_Test_Part3_GroundTruth.csv')
    assert os.path.exists(truthFile)

    testFiles = sorted(os.listdir(testDir))
    if len(testFiles) != 1:
        raise ScoreException('Multiple files submitted. Only 1 CSV file should '
                             'be submitted.')
    testFile = os.path.join(testDir, testFiles[0])

    # Load all data from the test CSV file
    testRows = []
    with open(testFile, 'rU') as testFileObj:
        try:
            testReader = csv.DictReader(
                testFileObj,
                fieldnames=['image_id', 'melanoma', 'seborrheic_keratosis'])
            for rowNum, testRow in enumerate(testReader):
                # TODO: handle extra fields
                if len(testRow.keys()) < 3:
                    raise ScoreException('Row %d has an incorrect number of'
                                         'fields. Three fields are expected: '
                                         '<image_id>, <melanoma>, '
                                         '<seborrheic_keratosis>' % rowNum)

                if not testRow['image_id']:
                    raise ScoreException('Could not find an image ID in the '
                                         'first field of row %d.' % rowNum)

                try:
                    testRow['melanoma'] = float(testRow['melanoma'])
                    testRow['seborrheic_keratosis'] = \
                        float(testRow['seborrheic_keratosis'])
                except (ValueError, TypeError):
                    raise ScoreException('Could not parse one of the second or '
                                         'third fields for "%s" (row %d) as '
                                         'floating-point values.' %
                                         (testRow['image'], rowNum))
                if not ((0.0 <= testRow['melanoma'] <= 1.0) and
                        (0.0 <= testRow['seborrheic_keratosis'] <= 1.0)):
                    raise ScoreException('One of the confidence values for '
                                         '"%s" (row %d) is outside the range '
                                         '[0.0, 1.0].' %
                                         (testRow['image'], rowNum))

                testRows.append(testRow)
        except csv.Error as e:
            raise ScoreException('CSV file parsing failed because: "%s"' %
                                 str(e))

    # Load the ground truth CSV file, and merge it with the test results
    combinedRows = []
    with open(truthFile) as truthFileObj:
        truthReader = csv.DictReader(truthFileObj)
        for truthRow in truthReader:
            # Find the matching test result
            testRow = matchRowName(truthRow['image_id'], testRows)

            combinedRows.append({
                'image_id': truthRow['image_id'],
                'truth_value_mel': float(truthRow['melanoma']),
                'test_value_mel': testRow['melanoma'],
                'truth_value_sebk': float(truthRow['seborrheic_keratosis']),
                'test_value_sebk': testRow['seborrheic_keratosis'],
            })

    # Build the Numpy arrays for calculations
    truthValues = np.array([value['truth_value_mel'] for value in combinedRows])
    testValues = np.array([value['test_value_mel'] for value in combinedRows])

    # Compute accuracy, sensitivity, and specificity
    truthBinaryValues = truthValues > 0.5
    testBinaryValues = testValues > 0.5
    metrics = computeCommonMetrics(truthBinaryValues, testBinaryValues)

    # Compute average precision
    metrics.extend(computeAveragePrecisionMetrics(truthValues, testValues))

    # Compute AUC
    metrics.extend(computeAUCMetrics(truthValues, testValues))

    # Compute specificity at 95% sensitivity
    metrics.extend(computeSPECMetrics(truthValues, testValues, 0.95))

    # Compute specificity at 98% sensitivity
    metrics.extend(computeSPECMetrics(truthValues, testValues, 0.98))

    # Compute specificity at 99% sensitivity
    metrics.extend(computeSPECMetrics(truthValues, testValues, 0.99))

    # Only a single score can be computed for the entire dataset
    scores = [
        {
            'dataset': 'aggregate',
            'metrics': metrics
        }
    ]
    return scores
