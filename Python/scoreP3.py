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
import re

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
        if truthImageId in testValue['image_id']
    ]

    if not testValueCandidates:
        raise ScoreException('No matching submissions for: %s' % truthImageName)
    elif len(testValueCandidates) > 1:
        raise ScoreException('Multiple matching submissions for: %s' %
                        truthImageName)
    return testValueCandidates[0]


def scoreP3(truthDir, testDir):
    truthFile = next(
        os.path.join(truthDir, f)
        for f in os.listdir(truthDir)
        if re.match(r'ISIC-2017_Test_(?:Test|Validation)_GroundTruth\.csv', f)
    )

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
                                         (testRow['image_id'], rowNum))
                if not ((0.0 <= testRow['melanoma'] <= 1.0) and
                        (0.0 <= testRow['seborrheic_keratosis'] <= 1.0)):
                    raise ScoreException('One of the confidence values for '
                                         '"%s" (row %d) is outside the range '
                                         '[0.0, 1.0].' %
                                         (testRow['image_id'], rowNum))

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
                'truth_value_melanoma': float(truthRow['melanoma']),
                'test_value_melanoma': testRow['melanoma'],
                'truth_value_seborrheic_keratosis':
                    float(truthRow['seborrheic_keratosis']),
                'test_value_seborrheic_keratosis':
                    testRow['seborrheic_keratosis'],
            })

        metricsByCategory = {}
    for category in ['melanoma', 'seborrheic_keratosis']:
        metricsByCategory[category] = []

        # Build the Numpy arrays for calculations
        truthValues = np.array(
            [value['truth_value_%s' % category] for value in combinedRows])
        testValues = np.array(
            [value['test_value_%s' % category] for value in combinedRows])

        # Compute accuracy, sensitivity, and specificity
        truthBinaryValues = truthValues > 0.5
        testBinaryValues = testValues > 0.5
        metricsByCategory[category].extend(
            computeCommonMetrics(truthBinaryValues, testBinaryValues))

        # Compute average precision
        metricsByCategory[category].extend(
            computeAveragePrecisionMetrics(truthValues, testValues))

        # Compute specificity at 82% sensitivity
        metricsByCategory[category].extend(
            computeSPECMetrics(truthValues, testValues, 0.82))

        # Compute specificity at 89% sensitivity
        metricsByCategory[category].extend(
            computeSPECMetrics(truthValues, testValues, 0.89))

        # Compute specificity at 95% sensitivity
        metricsByCategory[category].extend(
            computeSPECMetrics(truthValues, testValues, 0.95))

        # Compute AUC
        metricsByCategory[category].extend(
            computeAUCMetrics(truthValues, testValues))

    # Compute mean metrics for combined melanoma and seborrheic_keratosis
    metricsByCategory['mean'] = []
    for metricMelanoma, metricSeborrheicKeratosis in zip(
            metricsByCategory['melanoma'],
            metricsByCategory['seborrheic_keratosis']):

        assert metricMelanoma['name'] == metricSeborrheicKeratosis['name']
        metricsByCategory['mean'].append({
            'name': metricMelanoma['name'],
            'value': (metricMelanoma['value'] +
                      metricSeborrheicKeratosis['value']) / 2.0
        })

    # Combine all metrics into a single list
    allMetrics = [
        {
            'name': '%s_%s' % (metric['name'], category),
            'value': metric['value']
        }
        for category in ['melanoma', 'seborrheic_keratosis', 'mean']
        for metric in metricsByCategory[category]
    ]

    # Only a single score can be computed for the entire dataset
    scores = [
        {
            'dataset': 'aggregate',
            'metrics': allMetrics
        }
    ]
    return scores
