# coding=utf-8

import os

from PIL import Image
import numpy as np

from scoreCommon import computeCommonMetrics, computeSimilarityMetrics


def loadImage(imagePath, rsize=-1):
    try:
        image = Image.open(imagePath)
    except:
        raise Exception('Could not decode image: %s' %
                        os.path.basename(imagePath))

    if image.mode != 'L':
        raise Exception('Image %s is not single-channel (grayscale).' %
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
            raise Exception('Image %s contains values other than 0 and 255.' %
                            os.path.basename(imagePath))
    else:
        raise Exception('Image %s contains values other than 0 and 255.' %
                        os.path.basename(imagePath))

    return image


def matchInputFile(truthFile, testDir):
    # truthFile ~= 'ISIC_0000003_Segmentation.png'
    truthFileId = truthFile.split('_')[1]
    for testFile in os.listdir(testDir):
        if truthFileId in testFile:
            testPath = os.path.join(testDir, testFile)
            return testPath

    raise Exception('No matching submission image for: %s' % truthFile)


def scoreP1Image(truthPath, testPath):
    truthImage = loadImage(truthPath)
    testImage = loadImage(testPath)

    if testImage.shape[0:2] != truthImage.shape[0:2]:
        raise Exception('Image %s has dimensions %s; expected %s.' %
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
        # truthFile ~= 'ISIC_0000003_Segmentation.png'
        try:
            testPath = matchInputFile(truthFile, testDir)
            truthPath = os.path.join(truthDir, truthFile)

            datasetName = truthFile.rsplit('_', 1)[0]
            metrics = scoreP1Image(truthPath, testPath)
        except Exception as e:
            # print(str(e), file=sys.stderr)
            # TODO: Don't fail completely
            raise

        scores.append({
            'dataset': datasetName,
            'metrics': metrics
        })

    return scores
