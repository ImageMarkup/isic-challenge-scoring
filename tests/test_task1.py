from isic_challenge_scoring import task1


def test_score(task1_truth_path, task1_prediction_path):
    assert task1.score(task1_truth_path, task1_prediction_path)
