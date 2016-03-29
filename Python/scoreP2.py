# coding=utf-8

import json
import os

import numpy as np

from scoreCommon import ScoreException, matchInputFile, \
    computeCommonMetrics, computeAveragePrecisionMetrics


_FEATURE_NAMES = ['globules', 'streaks']


def loadFeatures(featuresPath):
    try:
        with open(featuresPath) as featuresFileObj:
            features = json.load(featuresFileObj)
    except IOError:
        raise ScoreException('Internal error: error reading JSON file: %s' %
                             os.path.basename(featuresPath))
    except ValueError:
        raise ScoreException('Could not parse file "%s" as JSON.' %
                             os.path.basename(featuresPath))

    if not isinstance(features, dict):
        raise ScoreException('JSON file %s does not contain an Object '
                             '(key-value mapping) at the top-level.' % 
                             os.path.basename(featuresPath))

    for featureName in _FEATURE_NAMES:
        if featureName not in features:
            raise ScoreException('JSON file "%s" does not contain an element '
                                 'for feature "%s".' %
                                 (os.path.basename(featuresPath), featureName))

        if not isinstance(features[featureName], list):
            raise ScoreException('Feature "%s" in JSON file "%s" is not an '
                                 'Array.' %
                                 (featureName, os.path.basename(featuresPath)))

        try:
            features[featureName] = [
                float(superpixelValue)
                for superpixelValue in features[featureName]
            ]
        except ValueError:
            raise ScoreException('Array for feature "%s" in JSON file "%s" '
                                 'contains non-floating-point value(s).' %
                                 (featureName, os.path.basename(featuresPath)))

        for superpixelValue in features[featureName]:
            if not (0.0 <= superpixelValue <= 1.0):
                raise ScoreException('Array for feature "%s" in JSON file "%s" '
                                'contains a value outside the range '
                                '[0.0, 1.0].' %
                                (featureName, os.path.basename(featuresPath)))

    return features


def scoreP2(truthDir, testDir):

    scores = []
    featureAllTruthValues = {}
    featureAllTestValues = {}
    for truthFile in sorted(os.listdir(truthDir)):
        testPath = matchInputFile(truthFile, testDir)
        truthPath = os.path.join(truthDir, truthFile)

        truthFeatures = loadFeatures(truthPath)
        testFeatures = loadFeatures(testPath)

        for featureName in _FEATURE_NAMES:
            if len(testFeatures[featureName]) != \
               len(truthFeatures[featureName]):
                raise ScoreException('Array for feature "%s" in JSON file "%s" '
                                     'is length %d (expected length %d).' %
                                     (featureName, os.path.basename(testPath),
                                      len(testFeatures[featureName]),
                                      len(truthFeatures[featureName])))

            # Build the Numpy arrays for calculations
            truthValues = np.array(truthFeatures[featureName])
            testValues = np.array(testFeatures[featureName])

            # Compute accuracy, sensitivity, and specificity
            truthBinaryValues = truthValues > 0.5
            testBinaryValues = testValues > 0.5
            metrics = computeCommonMetrics(truthBinaryValues, testBinaryValues)

            # Insert null average precision to keep matrix the correct size
            metrics.append({
                'name': 'average_precision',
                'value': None
            })

            # Store binary values for later use for average precision
            featureAllTruthValues.setdefault(featureName, []).extend(
                truthFeatures[featureName])
            featureAllTestValues.setdefault(featureName, []).extend(
                testFeatures[featureName])

            # truthPath ~= '/.../ISIC_0000003.json'
            datasetName = os.path.splitext(os.path.basename(truthPath))[0]
            scores.append({
                'dataset': '%s_%s' % (datasetName, featureName),
                'metrics': metrics
            })

    # Compute metrics all images
    aggregateScores = []
    for featureName in _FEATURE_NAMES:
        allTruthValues = np.array(featureAllTruthValues[featureName])
        allTestValues = np.array(featureAllTestValues[featureName])

        # Compute accuracy, sensitivity, and specificity over all images
        allTruthBinaryValues = allTruthValues > 0.5
        allTestBinaryValues = allTestValues > 0.5
        aggregateMetrics = computeCommonMetrics(
            allTruthBinaryValues, allTestBinaryValues)

        # Compute average precision over all images
        aggregateMetrics.extend(
            computeAveragePrecisionMetrics(allTruthValues, allTestValues))

        aggregateScores.append({
            'dataset': 'aggregate_%s' % featureName,
            'metrics': aggregateMetrics
        })

    return aggregateScores + scores
