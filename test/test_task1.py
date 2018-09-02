import numpy as np
import pytest

from isic_challenge_scoring import task1

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


@pytest.mark.parametrize(
    'truthBinaryImage,predictionBinaryImage,'
    'thresholdJaccard,jaccard,dice,sensitivity,specificity,accuracy', [
        (truthBinaryImage, emptyOverlapPredictionBinaryImage,
         0.0, 0/4, 2*0/4, 0/4, 12/12, 12/16),
        (truthBinaryImage, noOverlapPredictionBinaryImage,
         0.0, 0/8, 2*0/8, 0/4, 8/12, 8/16),
        (truthBinaryImage, quarterOverlapPredictionBinaryImage,
         0.0, 1/7, 2*1/8, 1/4, 9/12, 10/16),
        (truthBinaryImage, halfOverlapPredictionBinaryImage,
         0.0, 2/6, 2*2/8, 2/4, 10/12, 12/16),
        (truthBinaryImage, halfFilledPredictionBinaryImage,
         0.0, 2/4, 2*2/6, 2/4, 12/12, 14/16),
        (truthBinaryImage, threeQuarterFilledPredictionBinaryImage,
         3/4, 3/4, 2*3/7, 3/4, 12/12, 15/16),
        (truthBinaryImage, truthBinaryImage,
         4/4, 4/4, 2*4/8, 4/4, 12/12, 16/16),
        (truthBinaryImage, oneExtraPredictionBinaryImage,
         4/5, 4/5, 2*4/9, 4/4, 11/12, 15/16),
    ]
)
def test_scoreImage(
        truthBinaryImage, predictionBinaryImage,
        thresholdJaccard, jaccard, dice, sensitivity, specificity, accuracy):
    metrics = task1.scoreImage(truthBinaryImage, predictionBinaryImage)

    assert metrics['threshold_jaccard'] == thresholdJaccard
    assert metrics['jaccard'] == jaccard
    assert metrics['dice'] == dice
    assert metrics['sensitivity'] == sensitivity
    assert metrics['specificity'] == specificity
    assert metrics['accuracy'] == accuracy

    # Some mathematical equalities that will always hold
    assert pytest.approx(dice) == (2 * jaccard) / (1.0 + jaccard)
    assert pytest.approx(jaccard) == dice / (2.0 - dice)
