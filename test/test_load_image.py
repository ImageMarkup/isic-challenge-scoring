# -*- coding: utf-8 -*-
import pathlib

import pytest

from isic_challenge_scoring import load_image
from isic_challenge_scoring.exception import ScoreException


@pytest.mark.parametrize('truthFile, correctImageId, correctAttributeId', [
    ('/foo/ISIC_1234567.png', 'ISIC_1234567', None),
    ('/ISIC_1234567_Segmentation.png', 'ISIC_1234567', None),
    ('/ISIC_1234567_attribute_streaks.png', 'ISIC_1234567', 'streaks'),
    ('/ISIC_1234567_attribute_milia_like_cyst.png', 'ISIC_1234567', 'milia_like_cyst'),
])
def test_parseImageId_valid(truthFile, correctImageId, correctAttributeId):
    imagePair = load_image.ImagePair(truthFile=pathlib.Path(truthFile))

    imagePair.parseImageId()

    assert imagePair.imageId == correctImageId
    assert imagePair.attributeId == correctAttributeId


@pytest.mark.parametrize('truthFile', [
    '/LICENSE.txt',
    '/ISIC_123456.png',
])
def test_parseImageId_invalid(truthFile):
    imagePair = load_image.ImagePair(truthFile=pathlib.Path(truthFile))

    with pytest.raises(ScoreException):
        imagePair.parseImageId()
