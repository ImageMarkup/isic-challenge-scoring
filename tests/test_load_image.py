# -*- coding: utf-8 -*-
import pathlib

import pytest

from isic_challenge_scoring import load_image
from isic_challenge_scoring.exception import ScoreException


@pytest.mark.parametrize(
    'truth_file, correct_image_id, correct_attribute_id',
    [
        ('/foo/ISIC_1234567.png', 'ISIC_1234567', None),
        ('/ISIC_1234567_Segmentation.png', 'ISIC_1234567', None),
        ('/ISIC_1234567_attribute_streaks.png', 'ISIC_1234567', 'streaks'),
        ('/ISIC_1234567_attribute_milia_like_cyst.png', 'ISIC_1234567', 'milia_like_cyst'),
    ],
)
def test_parse_image_id_valid(truth_file, correct_image_id, correct_attribute_id):
    image_pair = load_image.ImagePair(truth_file=pathlib.Path(truth_file))

    image_pair.parse_image_id()

    assert image_pair.image_id == correct_image_id
    assert image_pair.attribute_id == correct_attribute_id


@pytest.mark.parametrize('truth_file', ['/LICENSE.txt', '/ISIC_123456.png'])
def test_parse_image_id_invalid(truth_file):
    image_pair = load_image.ImagePair(truth_file=pathlib.Path(truth_file))

    with pytest.raises(ScoreException):
        image_pair.parse_image_id()
