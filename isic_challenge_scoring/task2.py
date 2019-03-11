# -*- coding: utf-8 -*-
import pathlib
from typing import Dict, List

import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import createBinaryConfusionMatrix, normalizeConfusionMatrix
from isic_challenge_scoring.scoreCommon import iterImagePairs


def score(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> List[Dict]:
    confusionMatrics = pd.DataFrame([
        createBinaryConfusionMatrix(
            truthBinaryValues=truthImage > 128,
            predictionBinaryValues=predictionImage > 128,
            name=truthFileId
        )
        # TODO: truthFileId needs to include attribute
        for truthImage, predictionImage, truthFileId in
        iterImagePairs(truthPath, predictionPath)
    ])

    # Normalize all values, since image sizes vary
    normalizedConfusionMatrics = confusionMatrics.apply(
        normalizeConfusionMatrix,
        axis='columns'
    )

    sumConfusionMatrix = normalizedConfusionMatrics.sum(axis='index')

    # TODO: per-attribute metrics

    return [
        {
            'dataset': 'micro_average',
            'metrics': [
                {
                    'name': 'jaccard',
                    'value': metrics.binaryJaccard(sumConfusionMatrix)
                },
                {
                    'name': 'dice',
                    'value': metrics.binaryDice(sumConfusionMatrix)
                },
            ]
        }
    ]
