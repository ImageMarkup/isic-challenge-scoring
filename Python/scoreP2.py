# coding=utf-8

import json
import os

import numpy as np

from scoreCommon import matchInputFile, \
    computeCommonMetrics, computeAveragePrecisionMetrics


_FEATURE_NAMES = ['globules', 'streaks']


def loadFeatures(featuresPath):
    try:
        with open(featuresPath) as f:
            features = json.load(f)
    except IOError:
        raise Exception('Internal error: error reading JSON file: %s'
                        % os.path.basename(featuresPath))
    except ValueError:
        # TODO: is this the right error type?
        raise Exception('Could not parse file "%s" as JSON.' %
                        os.path.basename(featuresPath))

    if not isinstance(features, dict):
        raise Exception('JSON file %s does not contain an Object '
                        '(key-value mapping) at the top-level.' %
                        os.path.basename(featuresPath))

    for featureName in _FEATURE_NAMES:
        if featureName not in features:
            raise Exception('JSON file "%s" does not contain an element for '
                            'feature "%s".' %
                            (os.path.basename(featuresPath), featureName))

        if not isinstance(features[featureName], list):
            raise Exception('Feature "%s" in JSON file "%s" is not an Array.' %
                            (featureName, os.path.basename(featuresPath)))

        try:
            features[featureName] = [
                float(superpixelValue)
                for superpixelValue in features[featureName]
            ]
        except ValueError:
            raise Exception('Array for feature "%s" in JSON file "%s" contains '
                            'non-floating-point value(s).' %
                            (featureName, os.path.basename(featuresPath)))

        for superpixelValue in features[featureName]:
            if not (0.0 <= superpixelValue <= 1.0):
                raise Exception('Array for feature "%s" in JSON file "%s" '
                                'contains a value outside the range '
                                '[0.0, 1.0].' %
                                (featureName, os.path.basename(featuresPath)))

    return features


def scoreP2Features(truthPath, testPath):
    truthFeatures = loadFeatures(truthPath)
    testFeatures = loadFeatures(testPath)

    scores = []
    for featureName in _FEATURE_NAMES:
        if len(testFeatures[featureName]) != len(truthFeatures[featureName]):
            raise Exception('Array for feature "%s" in JSON file "%s" is length'
                            ' %d (expected length %d).' %
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

        # Compute average precision
        metrics.extend(computeAveragePrecisionMetrics(truthValues, testValues))

        # truthPath ~= '/.../ISIC_0000003.json'
        datasetName = os.path.splitext(os.path.basename(truthPath))[0]
        scores.append({
            'dataset': '%s_%s' % (datasetName, featureName),
            'metrics': metrics
        })

    return scores


def scoreP2(truthDir, testDir):
    scores = []
    for truthFile in sorted(os.listdir(truthDir)):
        testPath = matchInputFile(truthFile, testDir)
        truthPath = os.path.join(truthDir, truthFile)

        scores.extend(scoreP2Features(truthPath, testPath))

    return scores
