import warnings

import numpy as np
import pandas as pd
from rdp import rdp
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

    tp_counts = pd.Series(confusion_matrix.diagonal(), index=categories)

    # These are equal to rows of the confusion matrix
    true_label_frequencies = _get_frequencies(truth_labels, weights, categories)

    balanced_accuracy = tp_counts.divide(true_label_frequencies).mean()
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
    if not (0 < sensitivity_threshold <= 1.0):
        raise Exception(f'Out of bounds sensitivity_threshold: {sensitivity_threshold}.')

    # Get the ROC curve points
    # TODO: We must have both some true and false instances in truthProbabilities
    fp_rates, tp_rates, thresholds = sklearn.metrics.roc_curve(
        truth_probabilities,
        prediction_probabilities,
        sample_weight=weights,
        drop_intermediate=False,
    )

    # Calling sklearn.metrics.roc_auc_score with max_fpr always applies the McClish correction,
    # which is a transform to normalize partial AUC values into the range [0.5, 1] (for a given FPR
    # interval): http://www.ncbi.nlm.nih.gov/pubmed/2668680
    # McClish-normalized partial AUC values may be a helpful metric to evaluate on their own,
    # but they are incompatible with the overall AUC, and SciKit learn (unlike R) does not
    # provide a flag to return the raw partial AUC, so just compute the desired metric directly

    # Search for the index along the curve where sensitivity_threshold (i.e. tp_rate threshold)
    # occurs
    # Since tp_rates is ordered, searchsorted provides better performance than np.argmax
    # Use side='left' to include any following points with exactly the target value
    threshold_index = tp_rates.searchsorted(sensitivity_threshold, side='left')

    # Take only the segment >= the value at threshold_index
    tp_rates_segment = tp_rates[threshold_index:]
    fp_rates_segment = fp_rates[threshold_index:]

    # Create an additional ROC point at exactly the threshold value
    tp_rate_threshold = sensitivity_threshold
    # It will be the case that fp_rate_threshold <= tp_rates[threshold_index]
    # Since tp_rates may have repeated values (which is disallowed by np.interp), use a 2-value
    # segment directly around the threshold value
    # If fp_rate_threshold < tp_rates[threshold_index], the 2-value segment needs to start from the
    # location at threshold_index-1, so that it straddles fp_rate_threshold
    # Even if fp_rate_threshold == tp_rates[threshold_index], a 2-value segment starting before
    # threshold_index is guaranteed to have no duplicates, as threshold_index is the left
    # side of any series of duplicates (since it was found with searchsorted(..., side='left'))
    fp_rate_threshold = np.interp(
        tp_rate_threshold,
        tp_rates[threshold_index - 1 : threshold_index + 1],
        fp_rates[threshold_index - 1 : threshold_index + 1],
    )

    # Prepend the point to the segment
    tp_rates_segment = np.insert(tp_rates_segment, 0, tp_rate_threshold)
    fp_rates_segment = np.insert(fp_rates_segment, 0, fp_rate_threshold)

    partial_auc = sklearn.metrics.auc(fp_rates_segment, tp_rates_segment)
    return partial_auc


def average_precision(
    truth_probabilities: pd.Series, prediction_probabilities: pd.Series, weights: pd.Series
) -> float:
    with warnings.catch_warnings():
        # sklearn.metrics.average_precision_score sometimes causes warnings internally, but they
        # appear to be harmless
        warnings.filterwarnings(
            'ignore', category=RuntimeWarning, message=r'^invalid value encountered in true_divide$'
        )
        ap = sklearn.metrics.average_precision_score(
            truth_probabilities, prediction_probabilities, sample_weight=weights
        )
    return ap


def roc(
    truth_probabilities: pd.Series, prediction_probabilities: pd.Series, weights: pd.Series
) -> pd.DataFrame:
    fprs, tprs, thresholds = sklearn.metrics.roc_curve(
        truth_probabilities, prediction_probabilities, sample_weight=weights
    )

    roc = pd.DataFrame({'fpr': fprs, 'tpr': tprs}, index=thresholds, columns=['fpr', 'tpr'])

    if len(fprs) > 100:
        # simplify line using Ramer-Douglas-Peucker algorithm if more than 100 points
        points = np.vstack((fprs, tprs)).T
        # a simple test reduced a roc curve of 2161 items to
        # epsilon 0      ... 660
        # epsilon 0.0001 ... 573
        # epsilon 0.0005 ... 344
        # epsilon 0.001  ... 197
        # epsilon 0.005  ...  17
        mask = rdp(points, return_mask=True, epsilon=0.001)
        roc = roc[mask]

    return roc
