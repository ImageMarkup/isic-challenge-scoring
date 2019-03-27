# -*- coding: utf-8 -*-
import pathlib
import re
from typing import Dict, ValuesView

import numpy as np
import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import createBinaryConfusionMatrix
from isic_challenge_scoring.exception import ScoreException
from isic_challenge_scoring.load_csv import excludeRows, parseCsv, sortRows, validateRows


CATEGORIES = pd.Index(['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
EXCLUDE_LABELS = ['ISIC_0035068']


def computeMetrics(truthFileStream, predictionFileStream) -> Dict[str, Dict[str, float]]:
    truthProbabilities = parseCsv(truthFileStream, CATEGORIES)
    predictionProbabilities = parseCsv(predictionFileStream, CATEGORIES)

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
