# -*- coding: utf-8 -*-
from isic_challenge_scoring import task3


def test_score(task3_truth_file_path, task3_prediction_file_path):
    assert task3.score(task3_truth_file_path, task3_prediction_file_path)
