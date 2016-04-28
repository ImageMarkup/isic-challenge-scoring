# coding=utf-8

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


def scoreP3(truthDir, testDir, phaseNum='3'):
    truthFile = os.path.join(
        truthDir, 'ISBI2016_ISIC_Part%s_Test_GroundTruth.csv' % phaseNum)
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
            testReader = csv.DictReader(testFileObj,
                                        fieldnames=['image', 'confidence'])
            for rowNum, testRow in enumerate(testReader):
                # TODO: handle extra fields
                if len(testRow.keys()) < 2:
                    raise ScoreException('Row %d has an incorrect number of'
                                         'fields. Two fields are expected: '
                                         '<image_id>, <malignant_confidence>' %
                                         rowNum)

                if not testRow['image']:
                    raise ScoreException('Could not find an image ID in the '
                                         'first field of row %d.' % rowNum)

                try:
                    testRow['confidence'] = float(testRow['confidence'])
                except (ValueError, TypeError):
                    raise ScoreException('Could not parse the second field for '
                                         '"%s" (row %d) as a floating-point '
                                         'value.' % (testRow['image'], rowNum))
                if not (0.0 <= testRow['confidence'] <= 1.0):
                    raise ScoreException('The confidence value for "%s" (row %d)'
                                         ' is outside the range [0.0, 1.0].' %
                                         (testRow['image'], rowNum))

                testRows.append(testRow)
        except csv.Error as e:
            raise ScoreException('CSV file parsing failed because: "%s"' %
                                 str(e))

    # Load the ground truth CSV file, and merge it with the test results
    combinedRows = []
    with open(truthFile) as truthFileObj:
        truthReader = csv.DictReader(truthFileObj,
                                     fieldnames=['image', 'confidence'])
        for truthRow in truthReader:
            # Find the matching test result
            testRow = matchRowName(truthRow['image'], testRows)

            combinedRows.append({
                'image': truthRow['image'],
                'truth_value': float(truthRow['confidence']),
                'test_value': testRow['confidence'],
            })

    # Build the Numpy arrays for calculations
    truthValues = np.array([value['truth_value'] for value in combinedRows])
    testValues = np.array([value['test_value'] for value in combinedRows])

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
