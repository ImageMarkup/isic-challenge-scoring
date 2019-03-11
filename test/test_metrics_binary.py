# -*- coding: utf-8 -*-
import numpy as np
import pytest

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import createBinaryConfusionMatrix

truthBinaryImage = np.array([
    [True, True, False, False],
    [True, True, False, False],
    [False, False, False, False],
    [False, False, False, False],
])
emptyOverlapPredictionBinaryImage = np.array([
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
    [False, False, False, False],
])
noOverlapPredictionBinaryImage = np.array([
    [False, False, False, False],
    [False, False, False, False],
    [False, False, True, True],
    [False, False, True, True],
])
quarterOverlapPredictionBinaryImage = np.array([
    [False, False, False, False],
    [False, True, True, False],
    [False, True, True, False],
    [False, False, False, False],
])
halfOverlapPredictionBinaryImage = np.array([
    [False, True, True, False],
    [False, True, True, False],
    [False, False, False, False],
    [False, False, False, False],
])
halfFilledPredictionBinaryImage = np.array([
    [False, True, False, False],
    [False, True, False, False],
    [False, False, False, False],
    [False, False, False, False],
])
threeQuarterFilledPredictionBinaryImage = np.array([
    [False, True, False, False],
    [True, True, False, False],
    [False, False, False, False],
    [False, False, False, False],
])
oneExtraPredictionBinaryImage = np.array([
    [True, True, False, False],
    [True, True, False, False],
    [False, False, True, False],
    [False, False, False, False],
])
filledPredictionBinaryImage = np.array([
    [True, True, True, True],
    [True, True, True, True],
    [True, True, True, True],
    [True, True, True, True],
])


@pytest.mark.parametrize('truthBinaryImage, predictionBinaryImage, correctValue', [
    (truthBinaryImage, emptyOverlapPredictionBinaryImage, 12 / 16),
    (truthBinaryImage, noOverlapPredictionBinaryImage, 8 / 16),
    (truthBinaryImage, quarterOverlapPredictionBinaryImage, 10 / 16),
    (truthBinaryImage, halfOverlapPredictionBinaryImage, 12 / 16),
    (truthBinaryImage, halfFilledPredictionBinaryImage, 14 / 16),
    (truthBinaryImage, threeQuarterFilledPredictionBinaryImage, 15 / 16),
    (truthBinaryImage, truthBinaryImage, 16 / 16),
    (truthBinaryImage, oneExtraPredictionBinaryImage, 15 / 16),
    (truthBinaryImage, filledPredictionBinaryImage, 4 / 16),
])
def test_binaryAccuracy(truthBinaryImage, predictionBinaryImage, correctValue):
    cm = createBinaryConfusionMatrix(truthBinaryImage, predictionBinaryImage)

    value = metrics.binaryAccuracy(cm)

    assert value == correctValue


@pytest.mark.parametrize('truthBinaryImage, predictionBinaryImage, correctValue', [
    (truthBinaryImage, emptyOverlapPredictionBinaryImage, 0 / 4),
    (truthBinaryImage, noOverlapPredictionBinaryImage, 0 / 4),
    (truthBinaryImage, quarterOverlapPredictionBinaryImage, 1 / 4),
    (truthBinaryImage, halfOverlapPredictionBinaryImage, 2 / 4),
    (truthBinaryImage, halfFilledPredictionBinaryImage, 2 / 4),
    (truthBinaryImage, threeQuarterFilledPredictionBinaryImage, 3 / 4),
    (truthBinaryImage, truthBinaryImage, 4 / 4),
    (truthBinaryImage, oneExtraPredictionBinaryImage, 4 / 4),
    (truthBinaryImage, filledPredictionBinaryImage, 4 / 4),
])
def test_binarySensitivity(truthBinaryImage, predictionBinaryImage, correctValue):
    cm = createBinaryConfusionMatrix(truthBinaryImage, predictionBinaryImage)

    value = metrics.binarySensitivity(cm)

    assert value == correctValue


@pytest.mark.parametrize('truthBinaryImage, predictionBinaryImage, correctValue', [
    (truthBinaryImage, emptyOverlapPredictionBinaryImage, 12 / 12),
    (truthBinaryImage, noOverlapPredictionBinaryImage, 8 / 12),
    (truthBinaryImage, quarterOverlapPredictionBinaryImage, 9 / 12),
    (truthBinaryImage, halfOverlapPredictionBinaryImage, 10 / 12),
    (truthBinaryImage, halfFilledPredictionBinaryImage, 12 / 12),
    (truthBinaryImage, threeQuarterFilledPredictionBinaryImage, 12 / 12),
    (truthBinaryImage, truthBinaryImage, 12 / 12),
    (truthBinaryImage, oneExtraPredictionBinaryImage, 11 / 12),
    (truthBinaryImage, filledPredictionBinaryImage, 0 / 12),
])
def test_binarySpecificity(truthBinaryImage, predictionBinaryImage, correctValue):
    cm = createBinaryConfusionMatrix(truthBinaryImage, predictionBinaryImage)

    value = metrics.binarySpecificity(cm)

    assert value == correctValue


@pytest.mark.parametrize('truthBinaryImage, predictionBinaryImage, correctValue', [
    (truthBinaryImage, emptyOverlapPredictionBinaryImage, 0 / 4),
    (truthBinaryImage, noOverlapPredictionBinaryImage, 0 / 8),
    (truthBinaryImage, quarterOverlapPredictionBinaryImage, 1 / 7),
    (truthBinaryImage, halfOverlapPredictionBinaryImage, 2 / 6),
    (truthBinaryImage, halfFilledPredictionBinaryImage, 2 / 4),
    (truthBinaryImage, threeQuarterFilledPredictionBinaryImage, 3 / 4),
    (truthBinaryImage, truthBinaryImage, 4 / 4),
    (truthBinaryImage, oneExtraPredictionBinaryImage, 4 / 5),
    (truthBinaryImage, filledPredictionBinaryImage, 4 / 16),
])
def test_binaryJaccard(truthBinaryImage, predictionBinaryImage, correctValue):
    cm = createBinaryConfusionMatrix(truthBinaryImage, predictionBinaryImage)

    value = metrics.binaryJaccard(cm)

    assert value == correctValue


@pytest.mark.parametrize('truthBinaryImage, predictionBinaryImage, correctValue', [
    (truthBinaryImage, emptyOverlapPredictionBinaryImage, 0.0),
    (truthBinaryImage, noOverlapPredictionBinaryImage, 0.0),
    (truthBinaryImage, quarterOverlapPredictionBinaryImage, 0.0),
    (truthBinaryImage, halfOverlapPredictionBinaryImage, 0.0),
    (truthBinaryImage, halfFilledPredictionBinaryImage, 0.0),
    (truthBinaryImage, threeQuarterFilledPredictionBinaryImage, 3 / 4),
    (truthBinaryImage, truthBinaryImage, 4 / 4),
    (truthBinaryImage, oneExtraPredictionBinaryImage, 4 / 5),
    (truthBinaryImage, filledPredictionBinaryImage, 0.0),
])
def test_binaryThresholdJaccard(truthBinaryImage, predictionBinaryImage, correctValue):
    cm = createBinaryConfusionMatrix(truthBinaryImage, predictionBinaryImage)

    value = metrics.binaryThresholdJaccard(cm)

    assert value == correctValue


@pytest.mark.parametrize('truthBinaryImage, predictionBinaryImage, correctValue', [
    (truthBinaryImage, emptyOverlapPredictionBinaryImage, 2 * 0 / 4),
    (truthBinaryImage, noOverlapPredictionBinaryImage, 2 * 0 / 8),
    (truthBinaryImage, quarterOverlapPredictionBinaryImage, 2 * 1 / 8),
    (truthBinaryImage, halfOverlapPredictionBinaryImage, 2 * 2 / 8),
    (truthBinaryImage, halfFilledPredictionBinaryImage, 2 * 2 / 6),
    (truthBinaryImage, threeQuarterFilledPredictionBinaryImage, 2 * 3 / 7),
    (truthBinaryImage, truthBinaryImage, 2 * 4 / 8),
    (truthBinaryImage, oneExtraPredictionBinaryImage, 2 * 4 / 9),
    (truthBinaryImage, filledPredictionBinaryImage, 2 * 4 / 20),
])
def test_binaryDice(truthBinaryImage, predictionBinaryImage, correctValue):
    cm = createBinaryConfusionMatrix(truthBinaryImage, predictionBinaryImage)

    value = metrics.binaryDice(cm)

    assert value == correctValue


@pytest.mark.parametrize('truthBinaryImage, predictionBinaryImage, correctValue', [
    (truthBinaryImage, emptyOverlapPredictionBinaryImage, 1.0),  # would be 0/0 in theory
    (truthBinaryImage, noOverlapPredictionBinaryImage, 0 / 4),
    (truthBinaryImage, quarterOverlapPredictionBinaryImage, 1 / 4),
    (truthBinaryImage, halfOverlapPredictionBinaryImage, 2 / 4),
    (truthBinaryImage, halfFilledPredictionBinaryImage, 2 / 2),
    (truthBinaryImage, threeQuarterFilledPredictionBinaryImage, 3 / 3),
    (truthBinaryImage, truthBinaryImage, 4 / 4),
    (truthBinaryImage, oneExtraPredictionBinaryImage, 4 / 5),
    (truthBinaryImage, filledPredictionBinaryImage, 4 / 16),
])
def test_binaryPpv(truthBinaryImage, predictionBinaryImage, correctValue):
    cm = createBinaryConfusionMatrix(truthBinaryImage, predictionBinaryImage)

    value = metrics.binaryPpv(cm)

    assert value == correctValue


@pytest.mark.parametrize('truthBinaryImage, predictionBinaryImage, correctValue', [
    (truthBinaryImage, emptyOverlapPredictionBinaryImage, 12 / 16),
    (truthBinaryImage, noOverlapPredictionBinaryImage, 8 / 12),
    (truthBinaryImage, quarterOverlapPredictionBinaryImage, 9 / 12),
    (truthBinaryImage, halfOverlapPredictionBinaryImage, 10 / 12),
    (truthBinaryImage, halfFilledPredictionBinaryImage, 12 / 14),
    (truthBinaryImage, threeQuarterFilledPredictionBinaryImage, 12 / 13),
    (truthBinaryImage, truthBinaryImage, 12 / 12),
    (truthBinaryImage, oneExtraPredictionBinaryImage, 11 / 11),
    (truthBinaryImage, filledPredictionBinaryImage, 1.0),  # would be 0/0 in theory
])
def test_binaryNpv(truthBinaryImage, predictionBinaryImage, correctValue):
    cm = createBinaryConfusionMatrix(truthBinaryImage, predictionBinaryImage)

    value = metrics.binaryNpv(cm)

    assert value == correctValue
