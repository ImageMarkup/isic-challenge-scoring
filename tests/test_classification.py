import pytest

from isic_challenge_scoring.classification import ClassificationScore, ValidationMetric


def test_score(classification_truth_file_path, classification_prediction_file_path):
    assert ClassificationScore.from_file(
        classification_truth_file_path, classification_prediction_file_path
    )


@pytest.mark.parametrize(
    'validation_metric',
    [ValidationMetric.AUC, ValidationMetric.BALANCED_ACCURACY, ValidationMetric.AVERAGE_PRECISION],
)
def test_score_validation_metric(
    classification_truth_file_path, classification_prediction_file_path, validation_metric
):
    score = ClassificationScore.from_file(
        classification_truth_file_path,
        classification_prediction_file_path,
        validation_metric=validation_metric,
    )
    assert isinstance(score.validation, float)
