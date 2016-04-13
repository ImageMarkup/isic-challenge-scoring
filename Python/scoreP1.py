# coding=utf-8

import os



from scoreCommon import ScoreException, matchInputFile, loadSegmentationImage, \
    convertToNumPyArray, computeCommonMetrics, computeSimilarityMetrics


def scoreP1Image(truthPath, testPath):
    truthImage = loadSegmentationImage(truthPath)
    testImage = loadSegmentationImage(testPath)

    truthImage = convertToNumPyArray(truthImage)
    testImage = convertToNumPyArray(testImage)

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

        # truthFile ~= 'ISIC_0000003_Segmentation.png'
        datasetName = truthFile.rsplit('_', 1)[0]
        metrics = scoreP1Image(truthPath, testPath)

        scores.append({
            'dataset': datasetName,
            'metrics': metrics
        })

    return scores
