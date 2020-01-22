from dataclasses import dataclass, field
import pathlib
import re
from typing import Generator, Match, Optional, Set

import numpy as np
from PIL import Image, UnidentifiedImageError

from isic_challenge_scoring.types import ScoreException


@dataclass
class ImagePair:
    truth_file: pathlib.Path
    truth_image: np.ndarray = field(init=False)
    prediction_file: pathlib.Path = field(init=False)
    prediction_image: np.ndarray = field(init=False)
    image_id: str = field(init=False)
    attribute_id: Optional[str] = field(default=None, init=False)

    def parse_image_id(self) -> None:
        image_id_match: Optional[Match[str]] = re.search(r'ISIC_[0-9]{7}', self.truth_file.stem)
        if not image_id_match:
            raise Exception(f'Unknown ground truth file: {self.truth_file.name}.')
        self.image_id = image_id_match.group(0)

        attribute_id_match: Optional[Match[str]] = re.search(
            r'attribute_([a-z_]+)', self.truth_file.stem
        )
        if attribute_id_match:
            self.attribute_id = attribute_id_match.group(1)

    def find_prediction_file(self, prediction_path: pathlib.Path) -> None:
        image_number: str = self.image_id.split('_')[1]

        if not self.attribute_id:
            prediction_file_candidates = [
                prediction_file
                for prediction_file in prediction_path.iterdir()
                if image_number in prediction_file.stem
            ]
        else:
            prediction_file_candidates = [
                prediction_file
                for prediction_file in prediction_path.iterdir()
                if image_number in prediction_file.stem
                and self.attribute_id in prediction_file.stem
            ]

        if not prediction_file_candidates:
            raise ScoreException(f'No matching submission for: {self.truth_file.name}')
        elif len(prediction_file_candidates) > 1:
            raise ScoreException(f'Multiple matching submissions for: {self.truth_file.name}')

        self.prediction_file = prediction_file_candidates[0]

    def load_truth_image(self) -> None:
        self.truth_image = load_segmentation_image(self.truth_file)
        # TODO: Validate all ground truth as binary before upload
        # self.truth_image = assert_binary_image(self.truth_image, self.truth_file)

    def load_prediction_image(self) -> None:
        self.prediction_image = load_segmentation_image(self.prediction_file)
        if self.prediction_image.shape[0:2] != self.truth_image.shape[0:2]:
            raise ScoreException(
                f'Image {self.prediction_file.name} has dimensions '
                f'{self.prediction_image.shape[0:2]}; expected {self.truth_image.shape[0:2]}.'
            )


def load_segmentation_image(image_path: pathlib.Path) -> np.ndarray:
    """Load a segmentation image as a NumPy array, given a file path."""
    try:
        with Image.open(image_path) as image:
            # Ensure the image is loaded, sometimes NumPy fails to get the "__array_interface__"
            image.load()

            if image.mode == '1':
                # NumPy crashes if a 1-bit (black and white) image is directly
                # coerced to an array
                image = image.convert('L')

            if image.mode != 'L':
                raise ScoreException(f'Image {image_path.name} is not single-channel (greyscale).')

            np_image = np.asarray(image)

    except UnidentifiedImageError:
        raise ScoreException(f'Could not decode image "{image_path.name}"')

    return np_image


def assert_binary_image(image: np.ndarray, image_path: pathlib.Path) -> np.ndarray:
    """Ensure a NumPy array image is binary, correcting if possible."""
    image_values: Set[int] = set(np.unique(image))
    if image_values <= {0, 255}:
        # Expected values
        pass
    elif len(image_values) <= 2:
        # Binary image with high value other than 255 can be corrected
        high_value = (image_values - {0}).pop()
        image /= high_value
        image *= 255
        if set(np.unique(image)) > {0, 255}:
            raise ScoreException(f'Image {image_path.name} contains values other than 0 and 255.')
    else:
        raise ScoreException(f'Image {image_path.name} contains values other than 0 and 255.')

    return image


def iter_image_pairs(
    truth_path: pathlib.Path, prediction_path: pathlib.Path
) -> Generator[ImagePair, None, None]:
    for truth_file in sorted(truth_path.iterdir()):
        if truth_file.name in {'ATTRIBUTION.txt', 'LICENSE.txt'}:
            continue

        image_pair = ImagePair(truth_file=truth_file)
        image_pair.parse_image_id()
        image_pair.find_prediction_file(prediction_path)
        image_pair.load_truth_image()
        image_pair.load_prediction_image()

        yield image_pair
