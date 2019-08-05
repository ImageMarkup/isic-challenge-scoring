# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from isic_challenge_scoring import metrics


def test_to_labels(categories):
    probabilities = pd.DataFrame(
        [
            # NV
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            # undecided
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # AKIEC
            [0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2],
            # undecided
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            # MEL
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        columns=categories,
    )

    labels = metrics._to_labels(probabilities)

    assert labels.equals(pd.Series(['NV', 'undecided', 'AKIEC', 'undecided', 'MEL']))


def test_get_frequencies(categories):
    labels = pd.Series(['MEL', 'MEL', 'VASC', 'AKIEC', 'MEL'])
    weights = pd.Series([1.0, 1.0, 1.0, 1.0, 0.0])

    label_frequencies = metrics._get_frequencies(labels, weights, categories)

    assert label_frequencies.equals(
        pd.Series(
            {'MEL': 2.0, 'NV': 0.0, 'BCC': 0.0, 'AKIEC': 1.0, 'BKL': 0.0, 'DF': 0.0, 'VASC': 1.0}
        )
    )
    # Ensure the ordering is correct (although Python3.6 dicts are ordered)
    assert label_frequencies.index.equals(categories)


@pytest.mark.parametrize(
    'truth_labels, prediction_labels, correct_value',
    [
        (['MEL'], ['MEL'], 1.0),
        (['NV'], ['NV'], 1.0),
        (['NV'], ['MEL'], 0.0),
        (['MEL', 'MEL'], ['MEL', 'MEL'], 1.0),
        (['MEL', 'NV'], ['MEL', 'NV'], 1.0),
        (['MEL', 'NV'], ['MEL', 'MEL'], 0.5),
        (['MEL', 'NV', 'MEL'], ['MEL', 'MEL', 'MEL'], 0.5),
        (['MEL', 'NV', 'MEL', 'MEL'], ['MEL', 'MEL', 'MEL', 'MEL'], 0.5),
        (['MEL', 'NV', 'MEL', 'MEL'], ['MEL', 'MEL', 'MEL', 'NV'], 1 / 3),
        (['MEL', 'NV', 'MEL', 'MEL'], ['NV', 'MEL', 'NV', 'NV'], 0.0),
    ],
)
@pytest.mark.parametrize('test_weight_zero', [False, True])
def test_label_balanced_multiclass_accuracy(
    truth_labels, prediction_labels, correct_value, test_weight_zero, categories
):
    weights = [1.0] * len(truth_labels)

    if test_weight_zero:
        # Insert a final incorrect, but unweighted prediction
        truth_labels = truth_labels + ['MEL']
        prediction_labels = prediction_labels + ['NV']
        weights = weights + [0.0]

    value = metrics._label_balanced_multiclass_accuracy(
        pd.Series(truth_labels), pd.Series(prediction_labels), pd.Series(weights), categories
    )

    assert value == correct_value


@pytest.mark.parametrize(
    'truth_probabilities, prediction_probabilities, sensitivity_threshold, correct_value',
    [
        # This only checks some edge cases for sanity
        # Perfect predictor, tolerant threshold
        ([0.0, 0.0, 1.0, 1.0], [0.2, 0.4, 0.6, 0.8], 0.0, 1.0),
        # Perfect predictor, stringent threshold
        ([0.0, 0.0, 1.0, 1.0], [0.2, 0.4, 0.6, 0.8], 1.0, 1.0),
        # 50/50 predictor, tolerant threshold
        ([0.0, 0.0, 1.0, 1.0], [0.3, 0.7, 0.3, 0.7], 0.0, 0.5),
        # 50/50 predictor, stringent threshold
        ([0.0, 0.0, 1.0, 1.0], [0.3, 0.7, 0.3, 0.7], 1.0, 0.0),
        # Wrong predictor, tolerant threshold
        ([0.0, 0.0, 1.0, 1.0], [0.8, 0.6, 0.4, 0.2], 0.0, 0.0),
        # Wrong predictor, stringent threshold
        ([0.0, 0.0, 1.0, 1.0], [0.8, 0.6, 0.4, 0.2], 1.0, 0.0),
    ],
)
@pytest.mark.parametrize('test_weight_zero', [False, True])
def test_auc_above_sensitivity(
    truth_probabilities,
    prediction_probabilities,
    sensitivity_threshold,
    correct_value,
    test_weight_zero,
):
    weights = [1.0] * len(truth_probabilities)

    if test_weight_zero:
        # Insert a final incorrect, but unweighted prediction
        truth_probabilities = truth_probabilities + [1.0]
        prediction_probabilities = prediction_probabilities + [0.2]
        weights = weights + [0.0]

    value = metrics.auc_above_sensitivity(
        pd.Series(truth_probabilities),
        pd.Series(prediction_probabilities),
        pd.Series(weights),
        sensitivity_threshold,
    )

    assert value == correct_value


@pytest.mark.parametrize(
    'truth_probabilities, prediction_probabilities, correct_roc',
    [
        # This only checks some edge cases for sanity
        # Sklearn example
        (
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.4, 0.35, 0.8],
            [
                {'fpr': 0.0, 'tpr': 0.0, 'threshold': 1.8},
                {'fpr': 0.0, 'tpr': 0.5, 'threshold': 0.8},
                {'fpr': 0.5, 'tpr': 0.5, 'threshold': 0.4},
                {'fpr': 0.5, 'tpr': 1.0, 'threshold': 0.35},
                {'fpr': 1.0, 'tpr': 1.0, 'threshold': 0.1},
            ],
        ),
        # Perfect predictor
        (
            [0.0, 0.0, 1.0, 1.0],
            [0.2, 0.4, 0.6, 0.8],
            [
                {'fpr': 0.0, 'tpr': 0.0, 'threshold': 1.8},
                {'fpr': 0.0, 'tpr': 0.5, 'threshold': 0.8},
                {'fpr': 0.0, 'tpr': 1.0, 'threshold': 0.6},
                {'fpr': 1.0, 'tpr': 1.0, 'threshold': 0.2},
            ],
        ),
        # 50/50 predictor
        (
            [0.0, 0.0, 1.0, 1.0],
            [0.3, 0.7, 0.3, 0.7],
            [
                {'fpr': 0.0, 'tpr': 0.0, 'threshold': 1.7},
                {'fpr': 0.5, 'tpr': 0.5, 'threshold': 0.7},
                {'fpr': 1.0, 'tpr': 1.0, 'threshold': 0.3},
            ],
        ),
        # Wrong predictor
        (
            [0.0, 0.0, 1.0, 1.0],
            [0.8, 0.6, 0.4, 0.2],
            [
                {'fpr': 0.0, 'tpr': 0.0, 'threshold': 1.8},
                {'fpr': 0.5, 'tpr': 0.0, 'threshold': 0.8},
                {'fpr': 1.0, 'tpr': 0.0, 'threshold': 0.6},
                {'fpr': 1.0, 'tpr': 1.0, 'threshold': 0.2},
            ],
        ),
    ],
)
@pytest.mark.parametrize('test_weight_zero', [False, True])
def test_roc(truth_probabilities, prediction_probabilities, correct_roc, test_weight_zero):
    weights = [1.0] * len(truth_probabilities)

    if test_weight_zero:
        # Insert a final incorrect, but unweighted prediction
        truth_probabilities = truth_probabilities + [1.0]
        # To avoid including additional threshold points in the output (which would be correct, but
        # difficult to assert), insert one of the existing prediction probability values
        prediction_probabilities = prediction_probabilities + [prediction_probabilities[0]]
        weights = weights + [0.0]

    roc = metrics.roc(
        pd.Series(truth_probabilities), pd.Series(prediction_probabilities), pd.Series(weights)
    )

    assert roc == correct_roc
