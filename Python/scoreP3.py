# coding=utf-8

import csv
import os



def matchRowName(truthImageName, testValues):
    # truthImageName ~= 'ISIC_0000003'
    truthImageId = truthImageName.split('_')[1]
    for testValue in testValues:
        if truthImageId in testValue['image']:
            return testValue


def scoreP3Confidence(truthConfidence, testConfidence):
    # TODO
    return []


def scoreP3(truthDir, testDir):
    truthFile = os.path.join(truthDir,
                             'ISBI2016_ISIC_Part3_Test_GroundTruth.csv')
    assert os.path.exists(truthFile)

    testFiles = sorted(os.listdir(testDir))
    if len(testFiles) != 1:
        raise Exception('Multiple files submitted. Only 1 CSV file should be '
                        'submitted.')
    testFile = testFiles[0]

    with open(truthFile) as truthFileObj:
        truthReader = csv.DictReader(truthFileObj,
                                     ffieldnames=['image', 'confidence'])
        truthValues = list(truthReader)

    with open(testFile) as testFileObj:
        testReader = csv.DictReader(testFileObj,
                                    fieldnames=['image', 'confidence'])
        testValues = []
        for rowNum, row in enumerate(testReader):
            # TODO: handle extra fields
            if len(row.keys()) < 2:
                raise Exception('Row %d has an incorrect number of fields. Two '
                                'fields are expected: '
                                '<image_id>, <malignant_confidence>' % rowNum)

            if not row['image']:
                raise Exception('Could not find an image ID in the first field '
                                'of row %d.' % rowNum)

            try:
                row['confidence'] = float(row['confidence'])
            except ValueError:
                raise Exception('Could not parse the second field for "%s" '
                                '(row %d) as a floating-point value.' %
                                (row['image'], rowNum))
            if not (0.0 <= row['confidence'] <= 1.0):
                raise Exception('The confidence value for "%s" (row %d) is '
                                'outside the range [0.0, 1.0].' %
                                (row['image'], rowNum))

            testValues.append(row)

    scores = []
    for truthValue in truthValues:
        testValue = matchRowName(truthValue['image'], testValues)

        metrics = scoreP3Confidence(truthValue['confidence'],
                                    testValue['confidence'])

        scores.append({
            'dataset': truthValue['image'],
            'metrics': metrics
        })

    return scores
