# -*- coding: utf-8 -*-
import pathlib

import numpy as np
import pandas as pd
import pytest

from isic_challenge_scoring.confusion import create_binary_confusion_matrix
from isic_challenge_scoring.load_image import load_segmentation_image


data_dir = (pathlib.Path(__file__).parent / 'data').resolve()


@pytest.fixture
def categories() -> pd.Index:
    return pd.Index(['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])


@pytest.fixture
def task1_truth_path() -> pathlib.Path:
    return data_dir / 'task1' / 'groundtruth'


@pytest.fixture
def task1_prediction_path() -> pathlib.Path:
    return data_dir / 'task1' / 'prediction'


@pytest.fixture
def task2_truth_path() -> pathlib.Path:
    return data_dir / 'task2' / 'groundtruth'


@pytest.fixture
def task2_prediction_path() -> pathlib.Path:
    return data_dir / 'task2' / 'prediction'


@pytest.fixture
def task3_truth_file_path() -> pathlib.Path:
    return data_dir / 'task3' / 'groundtruth' / 'ISIC2018_Task3_GroundTruth.csv'


@pytest.fixture
def task3_prediction_file_path() -> pathlib.Path:
    return data_dir / 'task3' / 'prediction' / 'ISIC2018_Task3_prediction.csv'


@pytest.fixture
def real_truth_binary_values(task1_truth_path) -> np.ndarray:
    # TODO: don't hardcode this filename
    truth_file = task1_truth_path / 'ISIC_0000193_segmentation.png'
    truth_image = load_segmentation_image(truth_file)
    binary_truth_values = (truth_image > 128).ravel()
    return binary_truth_values


@pytest.fixture
def real_prediction_binary_values(task1_prediction_path) -> np.ndarray:
    # TODO: don't hardcode this filename
    prediction_file = task1_prediction_path / 'ISIC_0000193_segmentation_prediction.png'
    prediction_image = load_segmentation_image(prediction_file)
    binary_prediction_values = (prediction_image > 128).ravel()
    return binary_prediction_values


@pytest.fixture
def real_cm(real_truth_binary_values, real_prediction_binary_values) -> pd.Series:
    return create_binary_confusion_matrix(real_truth_binary_values, real_prediction_binary_values)
