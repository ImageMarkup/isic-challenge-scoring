# -*- coding: utf-8 -*-
from isic_challenge_scoring import task3


def test_score(task3_truth_path, task3_prediction_path):
    assert task3.score(task3_truth_path, task3_prediction_path)
