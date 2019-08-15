import pytest

from isic_challenge_scoring import task2


@pytest.mark.skip
def test_score(task2_truth_path, task2_prediction_path):
    # TODO: fix fixtures
    assert task2.score(task2_truth_path, task2_prediction_path)
