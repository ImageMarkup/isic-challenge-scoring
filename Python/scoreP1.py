# coding=utf-8

import os

from PIL import Image
import numpy as np

from scoreCommon import ScoreException, matchInputFile, \
    computeCommonMetrics, computeSimilarityMetrics


def loadImage(imagePath, rsize=-1):
    try:
        image = Image.open(imagePath)
    except Exception as e:
        raise ScoreException('Could not decode image "%s" because: "%s"' %
                             os.path.basename(imagePath), str(e))

    if image.mode == '1':
        # NumPy crashes if a 1-bit (black and white) image is directly
        # coerced to an array
        image = image.convert('L')

    if image.mode != 'L':
        raise ScoreException('Image %s is not single-channel (grayscale).' %
                             os.path.basename(imagePath))

    image = np.array(image)

    imageValues = set(np.unique(image))
    if imageValues <= {0, 255}:
        # Expected values
        pass
    elif len(imageValues) <= 2:
        # Binary image with high value other than 255 can be corrected
        highValue = (imageValues - {0}).pop()
        image /= highValue
        image *= 255
        if set(np.unique(image)) > {0, 255}:
            raise ScoreException('Image %s contains values other than 0 and '
                                 '255.' % os.path.basename(imagePath))
    else:
        raise ScoreException('Image %s contains values other than 0 and 255.' %
                             os.path.basename(imagePath))

    # TODO: resize image?

    return image


def scoreP1Image(truthPath, testPath):
    truthImage = loadImage(truthPath)
    testImage = loadImage(testPath)

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
