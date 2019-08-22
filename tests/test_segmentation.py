from isic_challenge_scoring.segmentation import SegmentationScore


def test_score(segmentation_truth_path, segmentation_prediction_path):
    assert SegmentationScore.from_dir(segmentation_truth_path, segmentation_prediction_path)
