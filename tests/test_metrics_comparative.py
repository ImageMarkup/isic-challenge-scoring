import pytest
import sklearn.metrics

from isic_challenge_scoring import metrics


def test_binary_accuracy_reference(
    real_cm, real_truth_binary_values, real_prediction_binary_values
):
    value = metrics.binary_accuracy(real_cm)
    reference_value = sklearn.metrics.accuracy_score(
        real_truth_binary_values, real_prediction_binary_values
    )

    assert value == pytest.approx(reference_value)


def test_binary_sensitivity_reference(
    real_cm, real_truth_binary_values, real_prediction_binary_values
):
    value = metrics.binary_sensitivity(real_cm)
    reference_value = sklearn.metrics.recall_score(
        real_truth_binary_values, real_prediction_binary_values
    )

    assert value == pytest.approx(reference_value)


def test_binary_jaccard_reference(real_cm, real_truth_binary_values, real_prediction_binary_values):
    value = metrics.binary_jaccard(real_cm)
    reference_value = sklearn.metrics.jaccard_score(
        real_truth_binary_values, real_prediction_binary_values
    )

    assert value == pytest.approx(reference_value)


def test_binary_dice_reference(real_cm, real_truth_binary_values, real_prediction_binary_values):
    value = metrics.binary_dice(real_cm)
    reference_value = sklearn.metrics.f1_score(
        real_truth_binary_values, real_prediction_binary_values
    )

    assert value == pytest.approx(reference_value)


def test_binary_ppv_reference(real_cm, real_truth_binary_values, real_prediction_binary_values):
    value = metrics.binary_ppv(real_cm)
    reference_value = sklearn.metrics.precision_score(
        real_truth_binary_values, real_prediction_binary_values
    )

    assert value == pytest.approx(reference_value)


def test_jaccard_dice_equality(real_cm):
    # Some mathematical equalities which will always hold
    jaccard = metrics.binary_jaccard(real_cm)
    dice = metrics.binary_dice(real_cm)

    assert dice == (2 * jaccard) / (1.0 + jaccard)
    assert jaccard == dice / (2.0 - dice)
