# -*- coding: utf-8 -*-
import pathlib
import re
from typing import KeysView

import numpy as np
import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import create_binary_confusion_matrix
from isic_challenge_scoring.load_csv import parse_csv, parse_truth_csv, sort_rows, validate_rows
from isic_challenge_scoring.types import ScoreException, ScoresType


def compute_metrics(truth_file_stream, prediction_file_stream) -> ScoresType:
    truth_probabilities, truth_weights = parse_truth_csv(truth_file_stream)
    categories = truth_probabilities.columns
    prediction_probabilities = parse_csv(prediction_file_stream, categories)

    validate_rows(truth_probabilities, prediction_probabilities)

    sort_rows(truth_probabilities)
    sort_rows(prediction_probabilities)

    scores: ScoresType = {}
    for category in categories:
        truth_category_probabilities: pd.Series = truth_probabilities[category]
        prediction_category_probabilities: pd.Series = prediction_probabilities[category]

        truth_binary_values: pd.Series = truth_category_probabilities.gt(0.5)
        prediction_binary_values: pd.Series = prediction_category_probabilities.gt(0.5)

        category_cm = create_binary_confusion_matrix(
            truth_binary_values=truth_binary_values.to_numpy(),
            prediction_binary_values=prediction_binary_values.to_numpy(),
            weights=truth_weights.score_weight.to_numpy(),
            name=category,
        )

        scores[category] = {
            'accuracy': metrics.binary_accuracy(category_cm),
            'sensitivity': metrics.binary_sensitivity(category_cm),
            'specificity': metrics.binary_specificity(category_cm),
            'dice': metrics.binary_dice(category_cm),
            'ppv': metrics.binary_ppv(category_cm),
            'npv': metrics.binary_npv(category_cm),
            'auc': metrics.auc(
                truth_category_probabilities,
                prediction_category_probabilities,
                truth_weights.score_weight,
            ),
            'auc_sens_80': metrics.auc_above_sensitivity(
                truth_category_probabilities,
                prediction_category_probabilities,
                truth_weights.score_weight,
                0.80,
            ),
            'ap': metrics.average_precision(
                truth_category_probabilities,
                prediction_category_probabilities,
                truth_weights.score_weight,
            ),
            'roc': metrics.roc(
                truth_category_probabilities,
                prediction_category_probabilities,
                truth_weights.score_weight,
            ),
        }

    # Compute averages for all per-category metrics
    per_category_metrics: KeysView[str] = next(iter(scores.values())).keys()
    scores['macro_average'] = {
        metric: float(np.mean([scores[category][metric] for category in categories]))
        for metric in per_category_metrics
        if metric != 'roc'
    }

    # Compute multi-category aggregate metrics
    scores['aggregate'] = {
        'balanced_accuracy': metrics.balanced_multiclass_accuracy(
            truth_probabilities, prediction_probabilities, truth_weights.score_weight
        )
    }

    scores['overall'] = scores['aggregate']['balanced_accuracy']
    scores['validation'] = metrics.balanced_multiclass_accuracy(
        truth_probabilities, prediction_probabilities, truth_weights.validation_weight
    )

    return scores


def score(truth_path: pathlib.Path, prediction_path: pathlib.Path) -> ScoresType:
    for truth_file in truth_path.iterdir():
        if re.match(r'^ISIC.*GroundTruth\.csv$', truth_file.name):
            break
    else:
        raise ScoreException('Internal error, truth file could not be found.')

    prediction_files = [
        prediction_file
        for prediction_file in prediction_path.iterdir()
        if prediction_file.suffix.lower() == '.csv'
    ]
    if len(prediction_files) > 1:
        raise ScoreException(
            'Multiple prediction files submitted. Exactly one CSV file should be submitted.'
        )
    elif len(prediction_files) < 1:
        raise ScoreException(
            'No prediction files submitted. Exactly one CSV file should be submitted.'
        )
    prediction_file = prediction_files[0]

    with truth_file.open('rb') as truth_file_stream, prediction_file.open(
        'rb'
    ) as prediction_file_stream:
        return compute_metrics(truth_file_stream, prediction_file_stream)
