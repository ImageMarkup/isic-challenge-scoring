import numpy as np
import pytest

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import create_binary_confusion_matrix

truth_binary_image = np.array(
    [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
)
empty_overlap_prediction_binary_image = np.array(
    [
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
)
no_overlap_prediction_binary_image = np.array(
    [
        [False, False, False, False],
        [False, False, False, False],
        [False, False, True, True],
        [False, False, True, True],
    ]
)
quarter_overlap_prediction_binary_image = np.array(
    [
        [False, False, False, False],
        [False, True, True, False],
        [False, True, True, False],
        [False, False, False, False],
    ]
)
half_overlap_prediction_binary_image = np.array(
    [
        [False, True, True, False],
        [False, True, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
)
half_filled_prediction_binary_image = np.array(
    [
        [False, True, False, False],
        [False, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
)
three_quarter_filled_prediction_binary_image = np.array(
    [
        [False, True, False, False],
        [True, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
)
one_extra_prediction_binary_image = np.array(
    [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, False, False],
    ]
)
filled_prediction_binary_image = np.array(
    [
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, True],
    ]
)


@pytest.mark.parametrize(
    'truth_binary_image, prediction_binary_image, correct_value',
    [
        (truth_binary_image, empty_overlap_prediction_binary_image, 12 / 16),
        (truth_binary_image, no_overlap_prediction_binary_image, 8 / 16),
        (truth_binary_image, quarter_overlap_prediction_binary_image, 10 / 16),
        (truth_binary_image, half_overlap_prediction_binary_image, 12 / 16),
        (truth_binary_image, half_filled_prediction_binary_image, 14 / 16),
        (truth_binary_image, three_quarter_filled_prediction_binary_image, 15 / 16),
        (truth_binary_image, truth_binary_image, 16 / 16),
        (truth_binary_image, one_extra_prediction_binary_image, 15 / 16),
        (truth_binary_image, filled_prediction_binary_image, 4 / 16),
    ],
)
def test_binary_accuracy(truth_binary_image, prediction_binary_image, correct_value):
    cm = create_binary_confusion_matrix(truth_binary_image, prediction_binary_image)

    value = metrics.binary_accuracy(cm)

    assert value == correct_value


@pytest.mark.parametrize(
    'truth_binary_image, prediction_binary_image, correct_value',
    [
        (truth_binary_image, empty_overlap_prediction_binary_image, 0 / 4),
        (truth_binary_image, no_overlap_prediction_binary_image, 0 / 4),
        (truth_binary_image, quarter_overlap_prediction_binary_image, 1 / 4),
        (truth_binary_image, half_overlap_prediction_binary_image, 2 / 4),
        (truth_binary_image, half_filled_prediction_binary_image, 2 / 4),
        (truth_binary_image, three_quarter_filled_prediction_binary_image, 3 / 4),
        (truth_binary_image, truth_binary_image, 4 / 4),
        (truth_binary_image, one_extra_prediction_binary_image, 4 / 4),
        (truth_binary_image, filled_prediction_binary_image, 4 / 4),
    ],
)
def test_binary_sensitivity(truth_binary_image, prediction_binary_image, correct_value):
    cm = create_binary_confusion_matrix(truth_binary_image, prediction_binary_image)

    value = metrics.binary_sensitivity(cm)

    assert value == correct_value


@pytest.mark.parametrize(
    'truth_binary_image, prediction_binary_image, correct_value',
    [
        (truth_binary_image, empty_overlap_prediction_binary_image, 12 / 12),
        (truth_binary_image, no_overlap_prediction_binary_image, 8 / 12),
        (truth_binary_image, quarter_overlap_prediction_binary_image, 9 / 12),
        (truth_binary_image, half_overlap_prediction_binary_image, 10 / 12),
        (truth_binary_image, half_filled_prediction_binary_image, 12 / 12),
        (truth_binary_image, three_quarter_filled_prediction_binary_image, 12 / 12),
        (truth_binary_image, truth_binary_image, 12 / 12),
        (truth_binary_image, one_extra_prediction_binary_image, 11 / 12),
        (truth_binary_image, filled_prediction_binary_image, 0 / 12),
    ],
)
def test_binary_specificity(truth_binary_image, prediction_binary_image, correct_value):
    cm = create_binary_confusion_matrix(truth_binary_image, prediction_binary_image)

    value = metrics.binary_specificity(cm)

    assert value == correct_value


@pytest.mark.parametrize(
    'truth_binary_image, prediction_binary_image, correct_value',
    [
        (truth_binary_image, empty_overlap_prediction_binary_image, 0 / 4),
        (truth_binary_image, no_overlap_prediction_binary_image, 0 / 8),
        (truth_binary_image, quarter_overlap_prediction_binary_image, 1 / 7),
        (truth_binary_image, half_overlap_prediction_binary_image, 2 / 6),
        (truth_binary_image, half_filled_prediction_binary_image, 2 / 4),
        (truth_binary_image, three_quarter_filled_prediction_binary_image, 3 / 4),
        (truth_binary_image, truth_binary_image, 4 / 4),
        (truth_binary_image, one_extra_prediction_binary_image, 4 / 5),
        (truth_binary_image, filled_prediction_binary_image, 4 / 16),
    ],
)
def test_binary_jaccard(truth_binary_image, prediction_binary_image, correct_value):
    cm = create_binary_confusion_matrix(truth_binary_image, prediction_binary_image)

    value = metrics.binary_jaccard(cm)

    assert value == correct_value


@pytest.mark.parametrize(
    'truth_binary_image, prediction_binary_image, correct_value',
    [
        (truth_binary_image, empty_overlap_prediction_binary_image, 0.0),
        (truth_binary_image, no_overlap_prediction_binary_image, 0.0),
        (truth_binary_image, quarter_overlap_prediction_binary_image, 0.0),
        (truth_binary_image, half_overlap_prediction_binary_image, 0.0),
        (truth_binary_image, half_filled_prediction_binary_image, 0.0),
        (truth_binary_image, three_quarter_filled_prediction_binary_image, 3 / 4),
        (truth_binary_image, truth_binary_image, 4 / 4),
        (truth_binary_image, one_extra_prediction_binary_image, 4 / 5),
        (truth_binary_image, filled_prediction_binary_image, 0.0),
    ],
)
def test_binary_threshold_jaccard(truth_binary_image, prediction_binary_image, correct_value):
    cm = create_binary_confusion_matrix(truth_binary_image, prediction_binary_image)

    value = metrics.binary_threshold_jaccard(cm)

    assert value == correct_value


@pytest.mark.parametrize(
    'truth_binary_image, prediction_binary_image, correct_value',
    [
        (truth_binary_image, empty_overlap_prediction_binary_image, 2 * 0 / 4),
        (truth_binary_image, no_overlap_prediction_binary_image, 2 * 0 / 8),
        (truth_binary_image, quarter_overlap_prediction_binary_image, 2 * 1 / 8),
        (truth_binary_image, half_overlap_prediction_binary_image, 2 * 2 / 8),
        (truth_binary_image, half_filled_prediction_binary_image, 2 * 2 / 6),
        (truth_binary_image, three_quarter_filled_prediction_binary_image, 2 * 3 / 7),
        (truth_binary_image, truth_binary_image, 2 * 4 / 8),
        (truth_binary_image, one_extra_prediction_binary_image, 2 * 4 / 9),
        (truth_binary_image, filled_prediction_binary_image, 2 * 4 / 20),
    ],
)
def test_binary_dice(truth_binary_image, prediction_binary_image, correct_value):
    cm = create_binary_confusion_matrix(truth_binary_image, prediction_binary_image)

    value = metrics.binary_dice(cm)

    assert value == correct_value


@pytest.mark.parametrize(
    'truth_binary_image, prediction_binary_image, correct_value',
    [
        (truth_binary_image, empty_overlap_prediction_binary_image, 1.0),  # would be 0/0 in theory
        (truth_binary_image, no_overlap_prediction_binary_image, 0 / 4),
        (truth_binary_image, quarter_overlap_prediction_binary_image, 1 / 4),
        (truth_binary_image, half_overlap_prediction_binary_image, 2 / 4),
        (truth_binary_image, half_filled_prediction_binary_image, 2 / 2),
        (truth_binary_image, three_quarter_filled_prediction_binary_image, 3 / 3),
        (truth_binary_image, truth_binary_image, 4 / 4),
        (truth_binary_image, one_extra_prediction_binary_image, 4 / 5),
        (truth_binary_image, filled_prediction_binary_image, 4 / 16),
    ],
)
def test_binary_ppv(truth_binary_image, prediction_binary_image, correct_value):
    cm = create_binary_confusion_matrix(truth_binary_image, prediction_binary_image)

    value = metrics.binary_ppv(cm)

    assert value == correct_value


@pytest.mark.parametrize(
    'truth_binary_image, prediction_binary_image, correct_value',
    [
        (truth_binary_image, empty_overlap_prediction_binary_image, 12 / 16),
        (truth_binary_image, no_overlap_prediction_binary_image, 8 / 12),
        (truth_binary_image, quarter_overlap_prediction_binary_image, 9 / 12),
        (truth_binary_image, half_overlap_prediction_binary_image, 10 / 12),
        (truth_binary_image, half_filled_prediction_binary_image, 12 / 14),
        (truth_binary_image, three_quarter_filled_prediction_binary_image, 12 / 13),
        (truth_binary_image, truth_binary_image, 12 / 12),
        (truth_binary_image, one_extra_prediction_binary_image, 11 / 11),
        (truth_binary_image, filled_prediction_binary_image, 1.0),  # would be 0/0 in theory
    ],
)
def test_binary_npv(truth_binary_image, prediction_binary_image, correct_value):
    cm = create_binary_confusion_matrix(truth_binary_image, prediction_binary_image)

    value = metrics.binary_npv(cm)

    assert value == correct_value
