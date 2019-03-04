# -*- coding: utf-8 -*-

###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import pathlib
import re
import warnings

import numpy as np
with warnings.catch_warnings():
    # See https://stackoverflow.com/a/40846742
    warnings.filterwarnings(
        'ignore',
        r'^numpy\.dtype size changed, may indicate binary incompatibility\.',
        RuntimeWarning)
    import pandas as pd
import sklearn.metrics  # noqa: E402

from .scoreCommon import ScoreException  # noqa: E402


CATEGORIES = pd.Index(['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
EXCLUDE_LABELS = ['ISIC_0035068']


def parseCsv(csvFileStream):
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


def excludeRows(probabilities: pd.DataFrame, labels: list):
    """Exclude rows with specified labels, in-place."""
    probabilities.drop(index=labels, inplace=True, errors='ignore')


def validateRows(truthProbabilities: pd.DataFrame, predictionProbabilities: pd.DataFrame):
    """
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


def toLabels(probabilities: pd.DataFrame) -> pd.Series:
    labels = probabilities.idxmax(axis='columns')

    # Find places where there are multiple maximum values
    maxProbabilities = probabilities.max(axis='columns')
    isMax: pd.DataFrame = probabilities.eq(maxProbabilities, axis='rows')
    numberOfMax: pd.Series = isMax.sum(axis='columns')
    multipleMax: pd.Series = numberOfMax.gt(1)
    # Set those locations as an 'undecided' label
    labels[multipleMax] = 'undecided'
    # TODO: emit a warning if any are set to 'undecided'

    return labels


def getFrequencies(labels: pd.Series) -> pd.Series:
    # .reindex sorts this by the order in CATEGORIES
    return labels.value_counts().reindex(CATEGORIES, fill_value=0)


def balancedMulticlassAccuracy(truthLabels: pd.Series, predictionLabels: pd.Series) -> float:
    # See http://scikit-learn.org/dev/modules/model_evaluation.html#balanced-accuracy-score ; in
    # summary, 'sklearn.metrics.balanced_accuracy_score' is for binary classification only, so we
    # need to implement our own; here, we implement a simpler version of "balanced accuracy" than
    # the definitions mentioned by SciKit learn, as it's just a normalization of TP scores by true
    # class proportions

    confusionMatrix = sklearn.metrics.confusion_matrix(
        truthLabels,
        predictionLabels,
        labels=CATEGORIES
    )
    # TODO: try to convert to a DataFrame, for useful debugging labels
    # confusionMatrix = pd.DataFrame(
    #     confusionMatrix,
    #     index=LABELS.map(lambda label: f'true_{label}'),
    #     columns=LABELS.map(lambda label: f'predicted_{label}')
    # )

    truePositiveCounts = pd.Series(confusionMatrix.diagonal(), index=CATEGORIES)

    # These are equal to rows of the confusion matrix
    trueLabelFrequencies = getFrequencies(truthLabels)

    balancedAccuracy = truePositiveCounts.divide(trueLabelFrequencies).mean()
    return balancedAccuracy


def computeMetrics(truthFileStream, predictionFileStream) -> list:
    truthProbabilities = parseCsv(truthFileStream)
    predictionProbabilities = parseCsv(predictionFileStream)

    excludeRows(truthProbabilities, EXCLUDE_LABELS)
    excludeRows(predictionProbabilities, EXCLUDE_LABELS)

    validateRows(truthProbabilities, predictionProbabilities)

    sortRows(truthProbabilities)
    sortRows(predictionProbabilities)

    truthLabels = toLabels(truthProbabilities)
    predictionLabels = toLabels(predictionProbabilities)

    scores = [
        {
            'dataset': 'aggregate',
            'metrics': [
                {
                    'name': 'balanced_accuracy',
                    'value': balancedMulticlassAccuracy(truthLabels, predictionLabels)
                }
            ]
        }
    ]
    return scores


def scoreP3(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> list:
    for truthFile in truthPath.iterdir():
        if re.match(r'^ISIC.*GroundTruth\.csv$', truthFile.name):
            break
    else:
        raise ScoreException('Internal error, truth file could not be found.')

    predictionFiles = list(
        predictionFile
        for predictionFile in predictionPath.iterdir()
        if predictionFile.suffix.lower() == '.csv'
    )
    if len(predictionFiles) > 1:
        raise ScoreException(
            'Multiple prediction files submitted. Exactly 1 CSV file should be submitted.')
    elif len(predictionFiles) < 1:
        raise ScoreException(
            'No prediction files submitted. Exactly 1 CSV file should be submitted.')
    predictionFile = predictionFiles[0]

    with truthFile.open('rb') as truthFileStream, predictionFile.open('rb') as predictionFileStream:
        return computeMetrics(truthFileStream, predictionFileStream)
