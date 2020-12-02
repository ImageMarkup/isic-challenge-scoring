import pathlib

import numpy as np
import pandas as pd
import pytest

from isic_challenge_scoring.confusion import create_binary_confusion_matrix
from isic_challenge_scoring.load_image import load_segmentation_image

data_dir = (pathlib.Path(__file__).parent / 'data').resolve()


@pytest.fixture
def test_images_path():
    return data_dir / 'images'


@pytest.fixture
def categories() -> pd.Index:
    return pd.Index(['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])


@pytest.fixture
def segmentation_truth_path() -> pathlib.Path:
    return data_dir / 'segmentation' / 'groundtruth'


@pytest.fixture
def segmentation_prediction_path() -> pathlib.Path:
    return data_dir / 'segmentation' / 'prediction'


@pytest.fixture
def task2_truth_path() -> pathlib.Path:
    return data_dir / 'task2' / 'groundtruth'


@pytest.fixture
def task2_prediction_path() -> pathlib.Path:
    return data_dir / 'task2' / 'prediction'


@pytest.fixture
def classification_truth_file_path() -> pathlib.Path:
    return data_dir / 'classification' / 'groundtruth' / 'ISIC2018_Task3_GroundTruth.csv'


@pytest.fixture
def classification_prediction_file_path() -> pathlib.Path:
    return data_dir / 'classification' / 'prediction' / 'ISIC2018_Task3_prediction.csv'


@pytest.fixture
def real_truth_binary_values(segmentation_truth_path) -> np.ndarray:
    # TODO: don't hardcode this filename
    truth_file = segmentation_truth_path / 'ISIC_0000193_segmentation.png'
    truth_image = load_segmentation_image(truth_file)
    binary_truth_values = (truth_image > 128).ravel()
    return binary_truth_values


@pytest.fixture
def real_prediction_binary_values(segmentation_prediction_path) -> np.ndarray:
    # TODO: don't hardcode this filename
    prediction_file = segmentation_prediction_path / 'ISIC_0000193_segmentation_prediction.png'
    prediction_image = load_segmentation_image(prediction_file)
    binary_prediction_values = (prediction_image > 128).ravel()
    return binary_prediction_values


@pytest.fixture
def real_cm(real_truth_binary_values, real_prediction_binary_values) -> pd.Series:
    return create_binary_confusion_matrix(real_truth_binary_values, real_prediction_binary_values)
