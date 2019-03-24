# -*- coding: utf-8 -*-
import pathlib
from typing import Dict, List

import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import createBinaryConfusionMatrix
from isic_challenge_scoring.load_image import iterImagePairs


def score(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> List[Dict]:
    confusionMatrics = pd.DataFrame([
        createBinaryConfusionMatrix(
            truthBinaryValues=imagePair.truthImage > 128,
            predictionBinaryValues=imagePair.predictionImage > 128,
            name=(imagePair.imageId, imagePair.attributeId)
        )
        for imagePair in iterImagePairs(truthPath, predictionPath)
    ])

    return [
        {
            'dataset': 'macro_average',
            'metrics': [
                {
                    'name': 'threshold_jaccard',
                    'value': confusionMatrics.apply(
                        metrics.binaryThresholdJaccard,
                        threshold=0.65,
                        axis='columns'
                    ).mean()
                },
                {
                    'name': 'jaccard',
                    'value': confusionMatrics.apply(
                        metrics.binaryJaccard,
                        axis='columns'
                    ).mean()
                },
                {
                    'name': 'dice',
                    'value': confusionMatrics.apply(
                        metrics.binaryDice,
                        axis='columns'
                    ).mean()
                },
                {
                    'name': 'accuracy',
                    'value': confusionMatrics.apply(
                        metrics.binaryAccuracy,
                        axis='columns'
                    ).mean()
                },
                {
                    'name': 'sensitivity',
                    'value': confusionMatrics.apply(
                        metrics.binarySensitivity,
                        axis='columns'
                    ).mean()
                },
                {
                    'name': 'specificity',
                    'value': confusionMatrics.apply(
                        metrics.binarySpecificity,
                        axis='columns'
                    ).mean()
                },
            ]
        }
    ]
