import pathlib
from typing import cast, Dict

import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import (
    create_binary_confusion_matrix,
    normalize_confusion_matrix,
)
from isic_challenge_scoring.load_image import iter_image_pairs


def score(truth_path: pathlib.Path, prediction_path: pathlib.Path):
    confusion_matrics = pd.DataFrame(
        [
            create_binary_confusion_matrix(
                truth_binary_values=image_pair.truth_image > 128,
                prediction_binary_values=image_pair.prediction_image > 128,
                name=(cast(str, image_pair.attribute_id), image_pair.image_id),
            )
            for image_pair in iter_image_pairs(truth_path, prediction_path)
        ]
    )
    confusion_matrics = confusion_matrics.reindex(
        index=pd.MultiIndex.from_tuples(confusion_matrics.index, names=('attribute_id', 'image_id'))
    )

    # Normalize all values, since image sizes vary
    normalized_confusion_matrics = confusion_matrics.apply(
        normalize_confusion_matrix, axis='columns'
    )

    scores: Dict = {}
    for attribute in sorted(confusion_matrics.index.unique('attribute_id')):
        attribute_confusion_matrics = normalized_confusion_matrics.loc(axis=0)[attribute, :]
        sum_attribute_confusion_matrics = attribute_confusion_matrics.sum(axis='index')

        scores[attribute] = {
            'jaccard': metrics.binary_jaccard(sum_attribute_confusion_matrics),
            'dice': metrics.binary_dice(sum_attribute_confusion_matrics),
        }

    sum_confusion_matrix = normalized_confusion_matrics.sum(axis='index')
    scores['micro_average'] = {
        'jaccard': metrics.binary_jaccard(sum_confusion_matrix),
        'dice': metrics.binary_dice(sum_confusion_matrix),
    }

    scores['overall'] = scores['micro_average']['jaccard']

    return scores
