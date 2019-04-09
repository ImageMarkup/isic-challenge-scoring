# -*- coding: utf-8 -*-
import pathlib
from typing import Dict

import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import create_binary_confusion_matrix
from isic_challenge_scoring.load_image import iter_image_pairs


def score(truth_path: pathlib.Path, prediction_path: pathlib.Path) -> Dict[str, Dict[str, float]]:
    confusion_matrics = pd.DataFrame(
        [
            create_binary_confusion_matrix(
                truth_binary_values=image_pair.truth_image > 128,
                prediction_binary_values=image_pair.predictionImage > 128,
                name=(image_pair.imageId, image_pair.attribute_id),
            )
            for image_pair in iter_image_pairs(truth_path, prediction_path)
        ]
    )

    return {
        'macro_average': {
            'threshold_jaccard': confusion_matrics.apply(
                metrics.binary_threshold_jaccard, threshold=0.65, axis='columns'
            ).mean(),
            'jaccard': confusion_matrics.apply(metrics.binary_jaccard, axis='columns').mean(),
            'dice': confusion_matrics.apply(metrics.binary_dice, axis='columns').mean(),
            'accuracy': confusion_matrics.apply(metrics.binary_accuracy, axis='columns').mean(),
            'sensitivity': confusion_matrics.apply(
                metrics.binary_sensitivity, axis='columns'
            ).mean(),
            'specificity': confusion_matrics.apply(
                metrics.binary_specificity, axis='columns'
            ).mean(),
        }
    }
