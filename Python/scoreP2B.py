# coding=utf-8

import os

import numpy as np
from PIL import Image
from collections import defaultdict

from scoreCommon import ScoreException, matchInputFile, loadSegmentationImage, \
    resizeImage, convertToNumPyArray, computeCommonMetrics, \
    computeSimilarityMetrics


def matchFeatureInputFile(truthFile, testDir):
    # truthFile ~= 'ISIC_0000003_globules.png'
    truthFileId, truthFileFeature = \
        os.path.splitext(truthFile)[0].split('_')[1:]

    testPathCandidates = [
        os.path.join(testDir, testFile)
        for testFile in os.listdir(testDir)
        if (truthFileId in testFile) and (truthFileFeature in testFile.lower())
    ]

    if not testPathCandidates:
        raise ScoreException('No matching submission for: %s' % truthFile)
    elif len(testPathCandidates) > 1:
        raise ScoreException('Multiple matching submissions for: %s' % truthFile)
    return testPathCandidates[0]


def scoreP2BConcatenatedImages(truthImage, testImage):
    truthBinaryImage = (truthImage > 128)
    testBinaryImage = (testImage > 128)

    metrics = computeCommonMetrics(truthBinaryImage, testBinaryImage)
    metrics.extend(computeSimilarityMetrics(truthBinaryImage, testBinaryImage))
    return metrics


def scoreP2B(truthDir, testDir):
    scores = []
    features = defaultdict(list)

    # Verify that test filenames match truth filenames
    for truthFile in sorted(os.listdir(truthDir)):
        testPath = matchFeatureInputFile(truthFile, testDir)
        truthPath = os.path.join(truthDir, truthFile)

        # truthFile ~= 'ISIC_0000003_globules.png'
        datasetName = os.path.splitext(truthFile)[0]

        featureName = datasetName.split('_')[2]
        features[featureName].append((truthPath, testPath))

    # Dimensions of resized images to add to concatenated arrays
    width = 512
    height = 512

    for featureName, paths in features.viewitems():
        # Pre-allocate concatenated arrays
        numFiles = len(paths)
        truthArray = np.zeros(numFiles * width * height, np.uint8)
        testArray = np.zeros(numFiles * width * height, np.uint8)

        for index, (truthPath, testPath) in enumerate(paths):
            # Load images and verify that dimensions match
            truthImage = loadSegmentationImage(truthPath)
            testImage = loadSegmentationImage(testPath)

            if testImage.size != truthImage.size:
                raise ScoreException('Image %s has dimensions %s; expected %s.' %
                                     (os.path.basename(testPath), testImage.size,
                                      truthImage.size))

            # Resize images
            truthImage = resizeImage(truthImage, width, height)
            testImage = resizeImage(testImage, width, height)

            # Convert images
            truthImage = convertToNumPyArray(truthImage, truthPath)
            testImage = convertToNumPyArray(testImage, testPath)

            # Add images to concatenated arrays
            truthArray[index * width * height : (index+1) * width * height] = truthImage.flatten()
            testArray[index * width * height : (index+1) * width * height] = testImage.flatten()

        # Score concatenated images
        metrics = scoreP2BConcatenatedImages(truthArray, testArray)

        scores.append({
            'dataset': 'collection_' + featureName,
            'metrics': metrics
        })

    return scores
