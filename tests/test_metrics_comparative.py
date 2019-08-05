# -*- coding: utf-8 -*-
import pytest
import sklearn.metrics

from isic_challenge_scoring import metrics


def test_binary_accuracy_reference(cm, truth_binary_values, prediction_binary_values):
    value = metrics.binary_accuracy(cm)
    reference_value = sklearn.metrics.accuracy_score(truth_binary_values, prediction_binary_values)

    assert value == pytest.approx(reference_value)


def test_binary_sensitivity_reference(cm, truth_binary_values, prediction_binary_values):
    value = metrics.binary_sensitivity(cm)
    reference_value = sklearn.metrics.recall_score(truth_binary_values, prediction_binary_values)

    assert value == pytest.approx(reference_value)


def test_binary_jaccard_reference(cm, truth_binary_values, prediction_binary_values):
    value = metrics.binary_jaccard(cm)
    reference_value = sklearn.metrics.jaccard_score(truth_binary_values, prediction_binary_values)

    assert value == pytest.approx(reference_value)


def test_binary_dice_reference(cm, truth_binary_values, prediction_binary_values):
    value = metrics.binary_dice(cm)
    reference_value = sklearn.metrics.f1_score(truth_binary_values, prediction_binary_values)

    assert value == pytest.approx(reference_value)


def test_binary_ppv_reference(cm, truth_binary_values, prediction_binary_values):
    value = metrics.binary_ppv(cm)
    reference_value = sklearn.metrics.precision_score(truth_binary_values, prediction_binary_values)

    assert value == pytest.approx(reference_value)


def test_jaccard_dice_equality(cm):
    # Some mathematical equalities which will always hold
    jaccard = metrics.binary_jaccard(cm)
    dice = metrics.binary_dice(cm)

    assert dice == (2 * jaccard) / (1.0 + jaccard)
    assert jaccard == dice / (2.0 - dice)
