# -*- coding: utf-8 -*-
from typing import Dict, List

import numpy as np
import pandas as pd
import sklearn.metrics


def _to_labels(probabilities: pd.DataFrame) -> pd.Series:
    labels = probabilities.idxmax(axis='columns')

    # Find places where there are multiple maximum values
    max_probabilities = probabilities.max(axis='columns')
    is_max: pd.DataFrame = probabilities.eq(max_probabilities, axis='rows')
    number_of_max: pd.Series = is_max.sum(axis='columns')
    multiple_max: pd.Series = number_of_max.gt(1)
    # Set those locations as an 'undecided' label
    labels[multiple_max] = 'undecided'
    # TODO: emit a warning if any are set to 'undecided'

    return labels


def _get_frequencies(labels: pd.Series, weights: pd.Series, categories: pd.Index) -> pd.Series:
    # Directly sum the weights, grouping them by label
    # .reindex sorts this by the order in categories
    return weights.groupby(labels, sort=False).sum().reindex(categories, fill_value=0)


def _label_balanced_multiclass_accuracy(
    truth_labels: pd.Series, prediction_labels: pd.Series, weights: pd.Series, categories: pd.Index
) -> float:
    # See http://scikit-learn.org/dev/modules/model_evaluation.html#balanced-accuracy-score ; in
    # summary, 'sklearn.metrics.balanced_accuracy_score' is for binary classification only, so we
    # need to implement our own; here, we implement a simpler version of "balanced accuracy" than
    # the definitions mentioned by SciKit learn, as it's just a normalization of TP scores by true
    # class proportions
    confusion_matrix = sklearn.metrics.confusion_matrix(
        truth_labels, prediction_labels, labels=categories, sample_weight=weights
    )
    # TODO: try to convert to a DataFrame, for useful debugging labels
    # confusion_matrix = pd.DataFrame(
    #     confusion_matrix,
    #     index=LABELS.map(lambda label: f'true_{label}'),
    #     columns=LABELS.map(lambda label: f'predicted_{label}')
    # )

    true_positive_counts = pd.Series(confusion_matrix.diagonal(), index=categories)

    # These are equal to rows of the confusion matrix
    true_label_frequencies = _get_frequencies(truth_labels, weights, categories)

    balanced_accuracy = true_positive_counts.divide(true_label_frequencies).mean()
    return balanced_accuracy


def balanced_multiclass_accuracy(
    truth_probabilities: pd.DataFrame, prediction_probabilities: pd.DataFrame, weights: pd.Series
) -> float:
    truth_labels = _to_labels(truth_probabilities)
    prediction_labels = _to_labels(prediction_probabilities)
    categories = truth_probabilities.columns

    # This is easier to test
    balanced_accuracy = _label_balanced_multiclass_accuracy(
        truth_labels, prediction_labels, weights, categories
    )
    return balanced_accuracy


def binary_accuracy(cm: pd.Series) -> float:
    return (cm.at['TP'] + cm.at['TN']) / (cm.at['TP'] + cm.at['TN'] + cm.at['FP'] + cm.at['FN'])


def binary_sensitivity(cm: pd.Series) -> float:
    if cm.at['TP'] + cm.at['FN'] == 0:
        # sensitivity can't be calculated if all are negative, so make this metric a freebie
        return 1.0
    else:
        return cm.at['TP'] / (cm.at['TP'] + cm.at['FN'])


def binary_specificity(cm: pd.Series) -> float:
    if cm.at['TN'] + cm.at['FP'] == 0:
        # specificity can't be calculated if all are positive, so make this metric a freebie
        return 1.0
    else:
        return cm.at['TN'] / (cm.at['TN'] + cm.at['FP'])


def binary_jaccard(cm: pd.Series) -> float:
    if cm.at['TP'] + cm.at['FP'] + cm.at['FN'] == 0:
        # Jaccard is ill-defined if all are negative and the prediction is perfect, but we'll
        # just score that as a perfect answer
        return 1.0
    else:
        return cm.at['TP'] / (cm.at['TP'] + cm.at['FP'] + cm.at['FN'])


def binary_threshold_jaccard(cm: pd.Series, threshold: float = 0.65) -> float:
    jaccard = binary_jaccard(cm)
    return jaccard if jaccard >= threshold else 0.0


def binary_dice(cm: pd.Series) -> float:
    if cm.at['TP'] + cm.at['FP'] + cm.at['FN'] == 0:
        # Dice is ill-defined if all are negative and the prediction is perfect, but we'll
        # just score that as a perfect answer
        return 1.0
    else:
        return (2 * cm.at['TP']) / ((2 * cm.at['TP']) + cm.at['FP'] + cm.at['FN'])


def binary_ppv(cm: pd.Series) -> float:
    if cm.at['TP'] + cm.at['FP'] == 0:
        # PPV is ill-defined if all predictions are negative; we'll score it as perfect, which
        # doesn't penalize the case where all are truly negative (a good predictor), and is sane
        # for the case where some are truly positive (a limitation of this metric)
        # Note, some other implementations would score the latter case as 0:
        # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        return 1.0
    else:
        return cm.at['TP'] / (cm.at['TP'] + cm.at['FP'])


def binary_npv(cm: pd.Series) -> float:
    if cm.at['TN'] + cm.at['FN'] == 0:
        # NPV is ill-defined if all predictions are positive; we'll score it as perfect, which
        # doesn't penalize the case where all are truly positive (a good predictor), and is sane
        # for the case where some are truly negative (a limitation of this metric)
        # Note, some other implementations would score the latter case as 0:
        # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        return 1.0
    else:
        return cm.at['TN'] / (cm.at['TN'] + cm.at['FN'])


def auc(
    truth_probabilities: pd.Series, prediction_probabilities: pd.Series, weights: pd.Series
) -> float:
    auc = sklearn.metrics.roc_auc_score(
        truth_probabilities, prediction_probabilities, sample_weight=weights
    )
    return auc


def auc_above_sensitivity(
    truth_probabilities: pd.Series,
    prediction_probabilities: pd.Series,
    weights: pd.Series,
    sensitivity_threshold: float,
) -> float:
    # Get the ROC curve points
    # TODO: We must have both some true and false instances in truthProbabilities
    false_positive_rates, true_positive_rates, thresholds = sklearn.metrics.roc_curve(
        truth_probabilities, prediction_probabilities, sample_weight=weights
    )

    # Search for the index along the curve where sensitivity_threshold occurs
    # sensitivity == true_positive_rate
    threshold_index = np.argmax(true_positive_rates >= sensitivity_threshold)

    # Get the corresponding FPR value at the threshold, needed for sklearn
    fpr_threshold = false_positive_rates[threshold_index]
    # Compute the partial AUC over the range [0, fpr_threshold], which is the complement of what we
    # want (i.e. [fpr_threshold, 1]), but this is all sklearn can provide
    if fpr_threshold == 0.0:
        # roc_auc_score's max_fpr requires a value > 0
        complementary_auc = 0.0
    else:
        complementary_auc = sklearn.metrics.roc_auc_score(
            truth_probabilities,
            prediction_probabilities,
            sample_weight=weights,
            max_fpr=fpr_threshold,
        )

    total_auc = sklearn.metrics.roc_auc_score(
        truth_probabilities, prediction_probabilities, sample_weight=weights
    )

    # complementary_auc is the left / lower area, and we want the right / upper area
    partial_auc = total_auc - complementary_auc
    return partial_auc


def average_precision(
    truth_probabilities: pd.Series, prediction_probabilities: pd.Series, weights: pd.Series
) -> float:
    ap = sklearn.metrics.average_precision_score(
        truth_probabilities, prediction_probabilities, sample_weight=weights
    )
    return ap


def roc(
    truth_probabilities: pd.Series, prediction_probabilities: pd.Series, weights: pd.Series
) -> List[Dict[str, float]]:
    fprs, tprs, thresholds = sklearn.metrics.roc_curve(
        truth_probabilities, prediction_probabilities, sample_weight=weights
    )
    roc = list(
        map(lambda fpr, tpr, t: {'fpr': fpr, 'tpr': tpr, 'threshold': t}, fprs, tprs, thresholds)
    )
    return roc
