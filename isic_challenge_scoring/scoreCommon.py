# -*- coding: utf-8 -*-
import pathlib
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score


class ScoreException(Exception):
    pass


def loadSegmentationImage(imagePath: pathlib.Path) -> np.ndarray:
    """Load a segmentation image as a NumPy array, given a file path."""
    try:
        image = Image.open(str(imagePath))
    except Exception as e:
        raise ScoreException(f'Could not decode image "{imagePath.name}" because: "{str(e)}"')

    if image.mode == '1':
        # NumPy crashes if a 1-bit (black and white) image is directly
        # coerced to an array
        image = image.convert('L')

    if image.mode != 'L':
        raise ScoreException(f'Image {imagePath.name} is not single-channel (greyscale).')

    image = np.array(image)

    return image


def assertBinaryImage(image, imageName):
    """Ensure a NumPy array image is binary, correcting if possible."""
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
            raise ScoreException(
                'Image %s contains values other than 0 and 255.' % imageName)
    else:
        raise ScoreException(
            'Image %s contains values other than 0 and 255.' % imageName)

    return image


def matchInputFile(truthFile: pathlib.Path, predictionPath: pathlib.Path) -> pathlib.Path:
    # TODO: refactor to reduce duplication
    if 'segmentation' in truthFile.stem.lower():
        # truthFile ~= 'ISIC_0000003_Segmentation.png'
        truthFileId = truthFile.stem.split('_')[1]
        predictionFileCandidates = [
            predictionFile
            for predictionFile in predictionPath.iterdir()
            if truthFileId in predictionFile.stem
        ]
    elif 'attribute' in truthFile.stem.lower():
        # truthFile ~= 'ISIC_0000003_attribute_streaks.png
        truthFileId = truthFile.stem.split('_')[1]
        truthFileAttribute = truthFile.stem.split('_')[3]
        predictionFileCandidates = [
            predictionFile
            for predictionFile in predictionPath.iterdir()
            if truthFileId in predictionFile.stem and truthFileAttribute in predictionFile.stem
        ]
    else:
        raise ScoreException(f'Internal error: unknown ground truth file: {truthFile.name}.')

    if not predictionFileCandidates:
        raise ScoreException(f'No matching submission for: {truthFile.name}')
    elif len(predictionFileCandidates) > 1:
        raise ScoreException(f'Multiple matching submissions for: {truthFile.name}')
    return predictionFileCandidates[0]


def iterImagePairs(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> \
        Tuple[np.ndarray, np.ndarray, str]:
    for truthFile in sorted(truthPath.iterdir()):
        if truthFile.name in {'ATTRIBUTION.txt', 'LICENSE.txt'}:
            continue

        predictionFile = matchInputFile(truthFile, predictionPath)

        truthImage = loadSegmentationImage(truthFile)
        truthImage = assertBinaryImage(truthImage, truthFile.name)

        predictionImage = loadSegmentationImage(predictionFile)
        if predictionImage.shape[0:2] != truthImage.shape[0:2]:
            raise ScoreException(
                f'Image {predictionFile.name} has dimensions {predictionImage.shape[0:2]}; '
                f'expected {truthImage.shape[0:2]}.')

        truthFileId = truthFile.stem.split('_')[1]

        yield truthImage, predictionImage, truthFileId


def computeAveragePrecisionMetrics(truthValues, testValues):
    """Compute average precision."""
    metrics = [
        {
            'name': 'average_precision',
            'value': average_precision_score(
                y_true=truthValues, y_score=testValues)
        }
    ]
    return metrics
