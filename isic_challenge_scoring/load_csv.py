# -*- coding: utf-8 -*-
from typing import TextIO, Tuple

import numpy as np
import pandas as pd

from isic_challenge_scoring.types import ScoreException


def parse_truth_csv(csv_file_stream: TextIO) -> Tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(csv_file_stream, header=0)

    table.set_index('image', drop=True, inplace=True, verify_integrity=False)

    # Support legacy truth files
    if 'score_weight' not in table.columns:
        table['score_weight'] = 1.0
    if 'validation_weight' not in table.columns:
        table['validation_weight'] = 1.0
    if 'ISIC_0035068' in table.index:
        # TODO: Move this to ground truth
        table.loc['ISIC_0035068', ['score_weight', 'validation_weight']] = 0.0

    probabilities = table.drop(columns=['score_weight', 'validation_weight'])
    weights = table[['score_weight', 'validation_weight']]

    return probabilities, weights


def parse_csv(csv_file_stream: TextIO, categories: pd.Index) -> pd.DataFrame:
    probabilities = pd.read_csv(csv_file_stream, header=0)

    if 'image' not in probabilities.columns:
        raise ScoreException('Missing column in CSV: "image".')

    probabilities.set_index('image', drop=True, inplace=True, verify_integrity=True)

    missing_columns = categories.difference(probabilities.columns)
    if not missing_columns.empty:
        raise ScoreException(f'Missing columns in CSV: {list(missing_columns)}.')

    extra_columns = probabilities.columns.difference(categories)
    if not extra_columns.empty:
        raise ScoreException(f'Extra columns in CSV: {list(extra_columns)}.')

    # sort by the order in categories
    probabilities = probabilities.reindex(categories, axis='columns')

    missing_rows = probabilities[probabilities.isnull().any(axis='columns')].index
    if not missing_rows.empty:
        raise ScoreException(f'Missing value(s) in CSV for images: {missing_rows.tolist()}.')

    non_float_columns = probabilities.dtypes[
        probabilities.dtypes.apply(lambda x: x != np.float64)
    ].index
    if not non_float_columns.empty:
        raise ScoreException(
            f'CSV contains non-floating-point value(s) in columns: {non_float_columns.tolist()}.'
        )
    # TODO: identify specific failed rows

    out_of_range_rows = probabilities[
        probabilities.applymap(lambda x: x < 0.0 or x > 1.0).any(axis='columns')
    ].index
    if not out_of_range_rows.empty:
        raise ScoreException(
            f'Values in CSV are outside the interval [0.0, 1.0] for images: '
            f'{out_of_range_rows.tolist()}.'
        )

    # TODO: fail on extra columns in data rows

    return probabilities


def validate_rows(truth_probabilities: pd.DataFrame, prediction_probabilities: pd.DataFrame):
    """
    Ensure prediction rows correspond to truth rows.

    Fail when predictionProbabilities is missing rows or has extra rows compared to
    truthProbabilities.
    """
    missing_images = truth_probabilities.index.difference(prediction_probabilities.index)
    if not missing_images.empty:
        raise ScoreException(f'Missing images in CSV: {missing_images.tolist()}.')

    extra_images = prediction_probabilities.index.difference(truth_probabilities.index)
    if not extra_images.empty:
        raise ScoreException(f'Extra images in CSV: {extra_images.tolist()}.')


def sort_rows(probabilities: pd.DataFrame):
    """Sort rows by labels, in-place."""
    probabilities.sort_index(axis='rows', inplace=True)
