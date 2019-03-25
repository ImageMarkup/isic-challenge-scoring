# -*- coding: utf-8 -*-
from dataclasses import dataclass
import pathlib
import re
from typing import Iterable, Match

import numpy as np
from PIL import Image

from isic_challenge_scoring.exception import ScoreException


@dataclass
class ImagePair:
    truthFile: pathlib.Path = None
    truthImage: np.ndarray = None
    predictionFile: pathlib.Path = None
    predictionImage: np.ndarray = None
    imageId: str = None
    attributeId: str = None

    def parseImageId(self):
        imageIdMatch: Match[str] = re.search(r'ISIC_[0-9]{7}', self.truthFile.stem)
        if not imageIdMatch:
            raise ScoreException(
                f'Internal error: unknown ground truth file: {self.truthFile.name}.')
        self.imageId = imageIdMatch.group(0)

        attributeIdMatch: Match[str] = re.search(r'attribute_([a-z_]+)', self.truthFile.stem)
        if attributeIdMatch:
            self.attributeId = attributeIdMatch.group(1)

    def findPredictionFile(self, predictionPath: pathlib.Path):
        imageNumber: str = self.imageId.split('_')[1]

        if not self.attributeId:
            predictionFileCandidates = [
                predictionFile
                for predictionFile in predictionPath.iterdir()
                if imageNumber in predictionFile.stem
            ]
        else:
            predictionFileCandidates = [
                predictionFile
                for predictionFile in predictionPath.iterdir()
                if imageNumber in predictionFile.stem and self.attributeId in predictionFile.stem
            ]

        if not predictionFileCandidates:
            raise ScoreException(f'No matching submission for: {self.truthFile.name}')
        elif len(predictionFileCandidates) > 1:
            raise ScoreException(f'Multiple matching submissions for: {self.truthFile.name}')

        self.predictionFile = predictionFileCandidates[0]

    def loadTruthImage(self):
        self.truthImage = loadSegmentationImage(self.truthFile)
        self.truthImage = assertBinaryImage(self.truthImage, self.truthFile)

    def loadPredictionImage(self):
        self.predictionImage = loadSegmentationImage(self.predictionFile)
        if self.predictionImage.shape[0:2] != self.truthImage.shape[0:2]:
            raise ScoreException(
                f'Image {self.predictionFile.name} has dimensions '
                f'{self.predictionImage.shape[0:2]}; expected {self.truthImage.shape[0:2]}.')


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


def assertBinaryImage(image: np.ndarray, imagePath: pathlib.Path):
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
                f'Image {imagePath.name} contains values other than 0 and 255.')
    else:
        raise ScoreException(
            f'Image {imagePath.name} contains values other than 0 and 255.')

    return image


def iterImagePairs(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> Iterable[ImagePair]:
    for truthFile in sorted(truthPath.iterdir()):
        if truthFile.name in {'ATTRIBUTION.txt', 'LICENSE.txt'}:
            continue

        imagePair = ImagePair(truthFile=truthFile)
        imagePair.parseImageId()
        imagePair.findPredictionFile(predictionPath)
        imagePair.loadTruthImage()
        imagePair.loadPredictionImage()

        yield imagePair
