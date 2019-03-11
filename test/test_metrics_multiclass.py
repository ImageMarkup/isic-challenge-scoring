# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from isic_challenge_scoring import metrics, task3


def test_toLabels():
    probabilities = pd.DataFrame([
        # NV
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        # undecided
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # AKIEC
        [0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2],
        # undecided
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        # MEL
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], columns=task3.CATEGORIES)

    labels = metrics._toLabels(probabilities)

    assert labels.equals(pd.Series([
        'NV',
        'undecided',
        'AKIEC',
        'undecided',
        'MEL'
    ]))


def test_getFrequencies():
    labels = pd.Series(['MEL', 'MEL', 'VASC', 'AKIEC'])

    labelFrequencies = metrics._getFrequencies(labels, task3.CATEGORIES)

    assert labelFrequencies.equals(pd.Series({
        'MEL': 2,
        'NV': 0,
        'BCC': 0,
        'AKIEC': 1,
        'BKL': 0,
        'DF': 0,
        'VASC': 1
    }))
    # Ensure the ordering is correct (although Python3.6 dicts are ordered)
    assert labelFrequencies.index.equals(task3.CATEGORIES)


@pytest.mark.parametrize('truthLabels, predictionLabels, correctValue', [
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
])
def test_labelBalancedMulticlassAccuracy(truthLabels, predictionLabels, correctValue):
    value = metrics._labelBalancedMulticlassAccuracy(
        pd.Series(truthLabels),
        pd.Series(predictionLabels),
        task3.CATEGORIES
    )

    assert value == correctValue


@pytest.mark.parametrize(
    'truthProbabilities, predictionProbabilities, sensitivityThreshold, correctValue', [
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
    ])
def test_aucAboveSensitivity(
        truthProbabilities, predictionProbabilities, sensitivityThreshold, correctValue):
    value = metrics.aucAboveSensitivity(
        pd.Series(truthProbabilities),
        pd.Series(predictionProbabilities),
        sensitivityThreshold
    )

    assert value == correctValue
