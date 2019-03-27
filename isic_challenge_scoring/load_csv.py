# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import pandas as pd

from isic_challenge_scoring.exception import ScoreException


def parseCsv(csvFileStream, categories: pd.Index) -> pd.DataFrame:
    probabilities = pd.read_csv(
        csvFileStream,
        header=0
    )

    if 'image' not in probabilities.columns:
        raise ScoreException('Missing column in CSV: "image".')

    probabilities.set_index('image', drop=True, inplace=True, verify_integrity=True)

    missingColumns = categories.difference(probabilities.columns)
    if not missingColumns.empty:
        raise ScoreException(f'Missing columns in CSV: {list(missingColumns)}.')

    extraColumns = probabilities.columns.difference(categories)
    if not extraColumns.empty:
        raise ScoreException(f'Extra columns in CSV: {list(extraColumns)}.')

    # sort by the order in categories
    probabilities = probabilities.reindex(categories, axis='columns')

    missingRows = probabilities[probabilities.isnull().any(axis='columns')].index
    if not missingRows.empty:
        raise ScoreException(f'Missing value(s) in CSV for images: {missingRows.tolist()}.')

    nonFloatColumns = probabilities.dtypes[probabilities.dtypes.apply(
        lambda x: x != np.float64
    )].index
    if not nonFloatColumns.empty:
        raise ScoreException(
            f'CSV contains non-floating-point value(s) in columns: {nonFloatColumns.tolist()}.')
    # TODO: identify specific failed rows

    outOfRangeRows = probabilities[probabilities.applymap(
        lambda x: x < 0.0 or x > 1.0
    ).any(axis='columns')].index
    if not outOfRangeRows.empty:
        raise ScoreException(
            f'Values in CSV are outside the interval [0.0, 1.0] for images: '
            f'{outOfRangeRows.tolist()}.')

    # TODO: fail on extra columns in data rows

    return probabilities


def excludeRows(probabilities: pd.DataFrame, labels: List):
    """Exclude rows with specified labels, in-place."""
    probabilities.drop(index=labels, inplace=True, errors='ignore')


def validateRows(truthProbabilities: pd.DataFrame, predictionProbabilities: pd.DataFrame):
    """
    Ensure prediction rows correspond to truth rows.

    Fail when predictionProbabilities is missing rows or has extra rows compared to
    truthProbabilities.
    """
    missingImages = truthProbabilities.index.difference(predictionProbabilities.index)
    if not missingImages.empty:
        raise ScoreException(f'Missing images in CSV: {missingImages.tolist()}.')

    extraImages = predictionProbabilities.index.difference(truthProbabilities.index)
    if not extraImages.empty:
        raise ScoreException(f'Extra images in CSV: {extraImages.tolist()}.')


def sortRows(probabilities: pd.DataFrame):
    """Sort rows by labels, in-place."""
    probabilities.sort_index(axis='rows', inplace=True)
