# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.metrics


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


def binaryAccuracy(cm: pd.Series) -> float:
    return (cm.at['TP'] + cm.at['TN']) / (cm.at['TP'] + cm.at['TN'] + cm.at['FP'] + cm.at['FN'])


def binarySensitivity(cm: pd.Series) -> float:
    if cm.at['TP'] + cm.at['FN'] == 0:
        # sensitivity can't be calculated if all are negative, so make this metric a freebie
        return 1.0
    else:
        return cm.at['TP'] / (cm.at['TP'] + cm.at['FN'])


def binarySpecificity(cm: pd.Series) -> float:
    if cm.at['TN'] + cm.at['FP'] == 0:
        # specificity can't be calculated if all are positive, so make this metric a freebie
        return 1.0
    else:
        return cm.at['TN'] / (cm.at['TN'] + cm.at['FP'])


def binaryJaccard(cm: pd.Series) -> float:
    if cm.at['TP'] + cm.at['FP'] + cm.at['FN'] == 0:
        # Jaccard is ill-defined if all are negative and the prediction is perfect, but we'll
        # just score that as a perfect answer
        return 1.0
    else:
        return cm.at['TP'] / (cm.at['TP'] + cm.at['FP'] + cm.at['FN'])


def binaryThresholdJaccard(cm: pd.Series, threshold: float = 0.65) -> float:
    jaccard = binaryJaccard(cm)
    return jaccard if jaccard >= threshold else 0.0


def binaryDice(cm: pd.Series) -> float:
    if cm.at['TP'] + cm.at['FP'] + cm.at['FN'] == 0:
        # Dice is ill-defined if all are negative and the prediction is perfect, but we'll
        # just score that as a perfect answer
        return 1.0
    else:
        return (2 * cm.at['TP']) / ((2 * cm.at['TP']) + cm.at['FP'] + cm.at['FN'])


def binaryPpv(cm: pd.Series) -> float:
    if cm.at['TP'] + cm.at['FP'] == 0:
        # PPV is ill-defined if all predictions are negative; we'll score it as perfect, which
        # doesn't penalize the case where all are truly negative (a good predictor), and is sane
        # for the case where some are truly positive (a limitation of this metric)
        # Note, some other implementations would score the latter case as 0:
        # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        return 1.0
    else:
        return cm.at['TP'] / (cm.at['TP'] + cm.at['FP'])


def binaryNpv(cm: pd.Series) -> float:
    if cm.at['TN'] + cm.at['FN'] == 0:
        # NPV is ill-defined if all predictions are positive; we'll score it as perfect, which
        # doesn't penalize the case where all are truly positive (a good predictor), and is sane
        # for the case where some are truly negative (a limitation of this metric)
        # Note, some other implementations would score the latter case as 0:
        # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        return 1.0
    else:
        return cm.at['TN'] / (cm.at['TN'] + cm.at['FN'])


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


def averagePrecision(truthProbabilities: pd.Series, predictionProbabilities: pd.Series) -> float:
    ap = sklearn.metrics.average_precision_score(truthProbabilities, predictionProbabilities)
    return ap
