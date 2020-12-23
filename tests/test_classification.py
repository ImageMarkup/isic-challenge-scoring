import pytest

from isic_challenge_scoring.classification import ClassificationMetric, ClassificationScore


@pytest.mark.parametrize(
    'target_metric',
    [
        ClassificationMetric.AUC,
        ClassificationMetric.BALANCED_ACCURACY,
        ClassificationMetric.AVERAGE_PRECISION,
    ],
)
def test_score(classification_truth_file_path, classification_prediction_file_path, target_metric):
    score = ClassificationScore.from_file(
        classification_truth_file_path,
        classification_prediction_file_path,
        target_metric,
    )
    assert isinstance(score.validation, float)
