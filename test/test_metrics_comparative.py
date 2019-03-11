# -*- coding: utf-8 -*-
import numpy as np
import pytest
import sklearn.metrics

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import createBinaryConfusionMatrix


def test_binaryAccuracy_reference(truth_binary_values, prediction_binary_values):
    cm = createBinaryConfusionMatrix(truth_binary_values, prediction_binary_values)

    value = metrics.binaryAccuracy(cm)
    reference_value = sklearn.metrics.accuracy_score(truth_binary_values, prediction_binary_values)

    assert value == pytest.approx(reference_value)


def test_binarySensitivity_reference(truth_binary_values, prediction_binary_values):
    cm = createBinaryConfusionMatrix(truth_binary_values, prediction_binary_values)

    value = metrics.binarySensitivity(cm)
    reference_value = sklearn.metrics.recall_score(truth_binary_values, prediction_binary_values)

    assert value == pytest.approx(reference_value)


def test_binaryJaccard_reference(truth_binary_values, prediction_binary_values):
    cm = createBinaryConfusionMatrix(truth_binary_values, prediction_binary_values)

    value = metrics.binaryJaccard(cm)
    # sklearn has a very idiosyncratic implementation of jaccard_similarity_score; unless the input
    # arrays are wrapped in an additional dimension, the result is actually the accuracy score
    # see: https://github.com/scikit-learn/scikit-learn/issues/3037
    reference_value = sklearn.metrics.jaccard_similarity_score(
        np.expand_dims(truth_binary_values, axis=0),
        np.expand_dims(prediction_binary_values, axis=0)
    )

    assert value == pytest.approx(reference_value)


def test_binaryDice_reference(truth_binary_values, prediction_binary_values):
    cm = createBinaryConfusionMatrix(truth_binary_values, prediction_binary_values)

    value = metrics.binaryDice(cm)
    reference_value = sklearn.metrics.f1_score(truth_binary_values, prediction_binary_values)

    assert value == pytest.approx(reference_value)


def test_binaryPpv_reference(truth_binary_values, prediction_binary_values):
    cm = createBinaryConfusionMatrix(truth_binary_values, prediction_binary_values)

    value = metrics.binaryPpv(cm)
    reference_value = sklearn.metrics.precision_score(truth_binary_values, prediction_binary_values)

    assert value == pytest.approx(reference_value)


def test_jaccard_dice_equality(truth_binary_values, prediction_binary_values):
    # Some mathematical equalities which will always hold
    cm = createBinaryConfusionMatrix(truth_binary_values, prediction_binary_values)

    jaccard = metrics.binaryJaccard(cm)
    dice = metrics.binaryDice(cm)

    assert dice == (2 * jaccard) / (1.0 + jaccard)
    assert jaccard == dice / (2.0 - dice)
