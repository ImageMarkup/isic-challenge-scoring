from isic_challenge_scoring.classification import ClassificationScore


def test_score(classification_truth_file_path, classification_prediction_file_path):
    assert ClassificationScore.from_file(
        classification_truth_file_path, classification_prediction_file_path
    )
