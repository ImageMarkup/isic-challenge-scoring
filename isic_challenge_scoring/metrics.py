# -*- coding: utf-8 -*-

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


def _toLabels(probabilities: pd.DataFrame) -> pd.Series:
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


def _getFrequencies(labels: pd.Series, categories: pd.Index) -> pd.Series:
    # .reindex sorts this by the order in categories
    return labels.value_counts().reindex(categories, fill_value=0)


def _labelBalancedMulticlassAccuracy(
        truthLabels: pd.Series, predictionLabels: pd.Series, categories: pd.Index) -> float:
    # See http://scikit-learn.org/dev/modules/model_evaluation.html#balanced-accuracy-score ; in
    # summary, 'sklearn.metrics.balanced_accuracy_score' is for binary classification only, so we
    # need to implement our own; here, we implement a simpler version of "balanced accuracy" than
    # the definitions mentioned by SciKit learn, as it's just a normalization of TP scores by true
    # class proportions
    confusionMatrix = sklearn.metrics.confusion_matrix(
        truthLabels,
        predictionLabels,
        labels=categories
    )
    # TODO: try to convert to a DataFrame, for useful debugging labels
    # confusionMatrix = pd.DataFrame(
    #     confusionMatrix,
    #     index=LABELS.map(lambda label: f'true_{label}'),
    #     columns=LABELS.map(lambda label: f'predicted_{label}')
    # )

    truePositiveCounts = pd.Series(confusionMatrix.diagonal(), index=categories)

    # These are equal to rows of the confusion matrix
    trueLabelFrequencies = _getFrequencies(truthLabels, categories)

    balancedAccuracy = truePositiveCounts.divide(trueLabelFrequencies).mean()
    return balancedAccuracy


def balancedMulticlassAccuracy(
        truthProbabilities: pd.DataFrame, predictionProbabilities: pd.DataFrame) -> float:
    truthLabels = _toLabels(truthProbabilities)
    predictionLabels = _toLabels(predictionProbabilities)
    categories = truthProbabilities.columns

    # This is easier to test
    balancedAccuracy = _labelBalancedMulticlassAccuracy(
        truthLabels, predictionLabels, categories)
    return balancedAccuracy


def binaryAccuracy(truthBinaryValues: pd.Series, predictionBinaryValues: pd.Series) -> float:
    accuracy = sklearn.metrics.accuracy_score(truthBinaryValues, predictionBinaryValues)
    return accuracy


def binarySensitivity(truthBinaryValues: pd.Series, predictionBinaryValues: pd.Series) -> float:
    if not truthBinaryValues.any():
        # sensitivity can't be calculated if all are negative, so make this metric a freebie
        sensitivity = 1.0
    else:
        sensitivity = sklearn.metrics.recall_score(truthBinaryValues, predictionBinaryValues)
    return sensitivity


def binarySpecificity(truthBinaryValues: pd.Series, predictionBinaryValues: pd.Series) -> float:
    if truthBinaryValues.all():
        # specificity can't be calculated if all are positive, so make this metric a freebie
        specificity = 1
    else:
        binaryConfusionMatrix = sklearn.metrics.confusion_matrix(
            truthBinaryValues,
            predictionBinaryValues
        )
        # TODO: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html to create a
        # binary confusion matrix type
        trueNegative, falsePositive, falseNegative, truePositive = binaryConfusionMatrix.ravel()

        specificity = trueNegative / (trueNegative + falsePositive)
    return specificity


def binaryF1(truthBinaryValues: pd.Series, predictionBinaryValues: pd.Series) -> float:
    f1 = sklearn.metrics.f1_score(truthBinaryValues, predictionBinaryValues)
    return f1


def binaryPpv(truthBinaryValues: pd.Series, predictionBinaryValues: pd.Series) -> float:
    ppv = sklearn.metrics.precision_score(truthBinaryValues, predictionBinaryValues)
    return ppv


def binaryNpv(truthBinaryValues: pd.Series, predictionBinaryValues: pd.Series) -> float:
    binaryConfusionMatrix = sklearn.metrics.confusion_matrix(
        truthBinaryValues,
        predictionBinaryValues
    )
    trueNegative, falsePositive, falseNegative, truePositive = binaryConfusionMatrix.ravel()

    npv = trueNegative / (trueNegative + falseNegative)
    return npv


def auc(truthProbabilities: pd.Series, predictionProbabilities: pd.Series) -> float:
    auc = sklearn.metrics.roc_auc_score(truthProbabilities, predictionProbabilities)
    return auc


def aucAboveSensitivity(truthProbabilities: pd.Series, predictionProbabilities: pd.Series,
                        sensitivityThreshold: float) -> float:
    # Get the ROC curve points
    # TODO: We must have both some true and false instances in truthProbabilities
    falsePositiveRates, truePositiveRates, thresholds = sklearn.metrics.roc_curve(
        truthProbabilities, predictionProbabilities)

    # Search for the index along the curve where sensitivityThreshold occurs
    # sensitivity == truePositiveRate
    thresholdIndex = np.argmax(truePositiveRates >= sensitivityThreshold)

    # Get the corresponding FPR value at the threshold, needed for sklearn
    fprThreshold = falsePositiveRates[thresholdIndex]
    # Compute the partial AUC over the range [0, fprThreshold], which is the complement of what we
    # want (i.e. [fprThreshold, 1]), but this is all sklearn can provide
    if fprThreshold == 0.0:
        # roc_auc_score's max_fpr requires a value > 0
        complementaryAuc = 0.0
    else:
        complementaryAuc = sklearn.metrics.roc_auc_score(
            truthProbabilities, predictionProbabilities,
            max_fpr=fprThreshold)

    totalAuc = sklearn.metrics.roc_auc_score(truthProbabilities, predictionProbabilities)

    # complementaryAuc is the left / lower area, and we want the right / upper area
    partialAuc = totalAuc - complementaryAuc
    return partialAuc
