# coding=utf-8

import os



from scoreCommon import ScoreException, matchInputFile, loadSegmentationImage, \
    computeCommonMetrics, computeSimilarityMetrics


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



def scoreP2BImage(truthPath, testPath):
    truthImage = loadSegmentationImage(truthPath)
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


def scoreP2B(truthDir, testDir):
    # Iterate over each file and call scoring executable on the pair
    scores = []
    for truthFile in sorted(os.listdir(truthDir)):
        testPath = matchFeatureInputFile(truthFile, testDir)
        truthPath = os.path.join(truthDir, truthFile)

        # truthFile ~= 'ISIC_0000003_globules.png'
        datasetName = os.path.splitext(truthFile)[0]
        metrics = scoreP2BImage(truthPath, testPath)

        scores.append({
            'dataset': datasetName,
            'metrics': metrics
        })

    return scores
