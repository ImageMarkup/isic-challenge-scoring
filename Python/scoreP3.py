# coding=utf-8

import csv
import os

import numpy as np
from sklearn.metrics import average_precision_score

from scoreCommon import computeCommonMetrics


def matchRowName(truthImageName, testValues):
    # truthImageName ~= 'ISIC_0000003'
    truthImageId = truthImageName.split('_')[1]
    for testValue in testValues:
        if truthImageId in testValue['image']:
            return testValue
    # TODO: ensure there's not a 2nd copy

    raise Exception('No matching submission image for: %s' % truthImageName)


def scoreP3(truthDir, testDir):
    truthFile = os.path.join(truthDir,
                             'ISBI2016_ISIC_Part3_Test_GroundTruth.csv')
    assert os.path.exists(truthFile)

    testFiles = sorted(os.listdir(testDir))
    if len(testFiles) != 1:
        raise Exception('Multiple files submitted. Only 1 CSV file should be '
                        'submitted.')
    testFile = testFiles[0]

    # Load all data from the test CSV file
    testRows = []
    with open(testFile) as testFileObj:
        testReader = csv.DictReader(testFileObj,
                                    fieldnames=['image', 'confidence'])
        for rowNum, testRow in enumerate(testReader):
            # TODO: handle extra fields
            if len(testRow.keys()) < 2:
                raise Exception('Row %d has an incorrect number of fields. Two '
                                'fields are expected: '
                                '<image_id>, <malignant_confidence>' % rowNum)

            if not testRow['image']:
                raise Exception('Could not find an image ID in the first field '
                                'of row %d.' % rowNum)

            try:
                testRow['confidence'] = float(testRow['confidence'])
            except ValueError:
                raise Exception('Could not parse the second field for "%s" '
                                '(row %d) as a floating-point value.' %
                                (testRow['image'], rowNum))
            if not (0.0 <= testRow['confidence'] <= 1.0):
                raise Exception('The confidence value for "%s" (row %d) is '
                                'outside the range [0.0, 1.0].' %
                                (testRow['image'], rowNum))

            testRows.append(testRow)

    # Load the ground truth CSV file, and merge it with the test results
    combinedRows = []
    with open(truthFile) as truthFileObj:
        truthReader = csv.DictReader(truthFileObj,
                                     ffieldnames=['image', 'confidence'])
        for truthRow in truthReader:
            # Find the matching test result
            testRow = matchRowName(truthRow['image'], testRows)

            combinedRows.append({
                'image': truthRow['image'],
                'truth_value': truthRow['confidence'],
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
    metrics.append({
        'name': 'average_precision',
        'value': average_precision_score(
            y_true=truthValues, y_score=testValues)
    })

    # Only a single score can be computed for the entire dataset
    scores = [
        {
            'dataset': 'aggregate',
            'metrics': metrics
        }
    ]
    return scores
