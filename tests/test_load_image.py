import pathlib

import pytest

from isic_challenge_scoring import ScoreException, load_image


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

    with pytest.raises(Exception):
        image_pair.parse_image_id()


@pytest.mark.parametrize(
    'test_image_name', ['binary.png', 'monochrome.png', 'monochrome_png_noext', 'monochrome.jpg']
)
def test_load_segmentation_image_valid(test_images_path, test_image_name):
    image_path = test_images_path / test_image_name
    np_image = load_image.load_segmentation_image(image_path)
    assert np_image is not None


@pytest.mark.parametrize('test_image_name', ['empty.png', 'random_bytes.png', 'rgb.png'])
def test_load_segmentation_image_invalid(test_images_path, test_image_name):
    image_path = test_images_path / test_image_name
    with pytest.raises(ScoreException):
        load_image.load_segmentation_image(image_path)
