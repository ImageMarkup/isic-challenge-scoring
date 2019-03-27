# -*- coding: utf-8 -*-
import pathlib
from typing import Dict

import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import createBinaryConfusionMatrix
from isic_challenge_scoring.load_image import iterImagePairs


def score(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> Dict[str, Dict[str, float]]:
    confusionMatrics = pd.DataFrame([
        createBinaryConfusionMatrix(
            truthBinaryValues=imagePair.truthImage > 128,
            predictionBinaryValues=imagePair.predictionImage > 128,
            name=(imagePair.imageId, imagePair.attributeId)
        )
        for imagePair in iterImagePairs(truthPath, predictionPath)
    ])

    return {
        'macro_average': {
            'threshold_jaccard': confusionMatrics.apply(
                metrics.binaryThresholdJaccard,
                threshold=0.65,
                axis='columns'
            ).mean(),
            'jaccard': confusionMatrics.apply(
                metrics.binaryJaccard,
                axis='columns'
            ).mean(),
            'dice': confusionMatrics.apply(
                metrics.binaryDice,
                axis='columns'
            ).mean(),
            'accuracy': confusionMatrics.apply(
                metrics.binaryAccuracy,
                axis='columns'
            ).mean(),
            'sensitivity': confusionMatrics.apply(
                metrics.binarySensitivity,
                axis='columns'
            ).mean(),
            'specificity': confusionMatrics.apply(
                metrics.binarySpecificity,
                axis='columns'
            ).mean(),
        }
    }
