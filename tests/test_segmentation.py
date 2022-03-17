import pytest

from isic_challenge_scoring.segmentation import SegmentationMetric, SegmentationScore


@pytest.mark.parametrize(
    'target_metric',
    [
        SegmentationMetric.JACCARD,
        SegmentationMetric.THRESHOLD_JACCARD,
        # SegmentationMetric.AUC,
    ],
)
def test_score(segmentation_truth_path, segmentation_prediction_path, target_metric):
    score = SegmentationScore.from_dir(
        segmentation_truth_path, segmentation_prediction_path, target_metric
    )
    assert isinstance(score.overall, float)
    assert isinstance(score.validation, float)
