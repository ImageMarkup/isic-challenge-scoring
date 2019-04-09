# -*- coding: utf-8 -*-
import pathlib

import pytest

from isic_challenge_scoring.load_image import load_segmentation_image


data_dir = (pathlib.Path(__file__).parent / 'data').resolve()


@pytest.fixture
def task1_truth_path():
    yield data_dir / 'task1' / 'groundtruth'


@pytest.fixture
def task1_prediction_path():
    yield data_dir / 'task1' / 'prediction'


@pytest.fixture
def task2_truth_path():
    yield data_dir / 'task2' / 'groundtruth'


@pytest.fixture
def task2_prediction_path():
    yield data_dir / 'task2' / 'prediction'


@pytest.fixture
def task3_truth_path():
    yield data_dir / 'task3' / 'groundtruth'


@pytest.fixture
def task3_prediction_path():
    yield data_dir / 'task3' / 'prediction'


@pytest.fixture
def truth_binary_values(task1_truth_path):
    # TODO: don't hardcode this filename
    truth_file = task1_truth_path / 'ISIC_0000193_segmentation.png'
    truth_image = load_segmentation_image(truth_file)
    binary_truth_values = (truth_image > 128).ravel()
    yield binary_truth_values


@pytest.fixture
def prediction_binary_values(task1_prediction_path):
    # TODO: don't hardcode this filename
    prediction_file = task1_prediction_path / 'ISIC_0000193_segmentation_prediction.png'
    prediction_image = load_segmentation_image(prediction_file)
    binary_prediction_values = (prediction_image > 128).ravel()
    yield binary_prediction_values
