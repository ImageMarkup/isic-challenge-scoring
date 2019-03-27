# -*- coding: utf-8 -*-
import pathlib
import re
from typing import Dict, List, ValuesView

import numpy as np
import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import createBinaryConfusionMatrix
from isic_challenge_scoring.exception import ScoreException


CATEGORIES = pd.Index(['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
EXCLUDE_LABELS = ['ISIC_0035068']


def parseCsv(csvFileStream) -> pd.DataFrame:
    probabilities = pd.read_csv(
        csvFileStream,
        header=0
    )

    if 'image' not in probabilities.columns:
        raise ScoreException('Missing column in CSV: "image".')

    probabilities.set_index('image', drop=True, inplace=True, verify_integrity=True)

    missingColumns = CATEGORIES.difference(probabilities.columns)
    if not missingColumns.empty:
        raise ScoreException(f'Missing columns in CSV: {list(missingColumns)}.')

    extraColumns = probabilities.columns.difference(CATEGORIES)
    if not extraColumns.empty:
        raise ScoreException(f'Extra columns in CSV: {list(extraColumns)}.')

    # sort by the order in CATEGORIES
    probabilities = probabilities.reindex(CATEGORIES, axis='columns')

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


def computeMetrics(truthFileStream, predictionFileStream) -> Dict[str, Dict[str, float]]:
    truthProbabilities = parseCsv(truthFileStream)
    predictionProbabilities = parseCsv(predictionFileStream)

    excludeRows(truthProbabilities, EXCLUDE_LABELS)
    excludeRows(predictionProbabilities, EXCLUDE_LABELS)

    validateRows(truthProbabilities, predictionProbabilities)

    sortRows(truthProbabilities)
    sortRows(predictionProbabilities)

    scores: Dict[str, Dict[str, float]] = {}
    for category in CATEGORIES:
        truthCategoryProbabilities: pd.Series = truthProbabilities[category]
        predictionCategoryProbabilities: pd.Series = predictionProbabilities[category]

        truthBinaryValues: pd.Series = truthCategoryProbabilities.gt(0.5)
        predictionBinaryValues: pd.Series = predictionCategoryProbabilities.gt(0.5)

        categoryCm = createBinaryConfusionMatrix(
            truthBinaryValues=truthBinaryValues.to_numpy(),
            predictionBinaryValues=predictionBinaryValues.to_numpy(),
            name=category
        )

        scores[category] = {
            'accuracy': metrics.binaryAccuracy(categoryCm),
            'sensitivity': metrics.binarySensitivity(categoryCm),
            'specificity': metrics.binarySpecificity(categoryCm),
            'dice': metrics.binaryDice(categoryCm),
            'ppv': metrics.binaryPpv(categoryCm),
            'npv': metrics.binaryNpv(categoryCm),
            'auc': metrics.auc(truthCategoryProbabilities, predictionCategoryProbabilities),
            'auc_sens_80': metrics.aucAboveSensitivity(
                truthCategoryProbabilities, predictionCategoryProbabilities, 0.80),
            'ap': metrics.averagePrecision(
                truthCategoryProbabilities, predictionCategoryProbabilities),
        }

    # Compute averages for all per-category metrics
    perCategoryMetrics: ValuesView[str] = next(iter(scores.values())).keys()
    for metric in perCategoryMetrics:
        scores['macro_average'][metric] = float(np.mean(
            scores[category][metric]
            for category in CATEGORIES
        ))

    # Compute multi-category aggregate metrics
    scores['aggregate'] = {
        'balanced_accuracy': metrics.balancedMulticlassAccuracy(
            truthProbabilities, predictionProbabilities)
    }

    return scores


def score(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> Dict[str, Dict[str, float]]:
    for truthFile in truthPath.iterdir():
        if re.match(r'^ISIC.*GroundTruth\.csv$', truthFile.name):
            break
    else:
        raise ScoreException('Internal error, truth file could not be found.')

    predictionFiles = [
        predictionFile
        for predictionFile in predictionPath.iterdir()
        if predictionFile.suffix.lower() == '.csv'
    ]
    if len(predictionFiles) > 1:
        raise ScoreException(
            'Multiple prediction files submitted. Exactly one CSV file should be submitted.')
    elif len(predictionFiles) < 1:
        raise ScoreException(
            'No prediction files submitted. Exactly one CSV file should be submitted.')
    predictionFile = predictionFiles[0]

    with truthFile.open('rb') as truthFileStream, predictionFile.open('rb') as predictionFileStream:
        return computeMetrics(truthFileStream, predictionFileStream)
