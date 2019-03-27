# -*- coding: utf-8 -*-
import pathlib
from typing import Dict

import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import createBinaryConfusionMatrix, normalizeConfusionMatrix
from isic_challenge_scoring.load_image import iterImagePairs


def score(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> Dict[str, Dict[str, float]]:
    confusionMatrics = pd.DataFrame([
        createBinaryConfusionMatrix(
            truthBinaryValues=imagePair.truthImage > 128,
            predictionBinaryValues=imagePair.predictionImage > 128,
            name=(imagePair.attributeId, imagePair.imageId)
        )
        for imagePair in iterImagePairs(truthPath, predictionPath)
    ])
    confusionMatrics = confusionMatrics.reindex(
        index=pd.MultiIndex.from_tuples(confusionMatrics.index, names=('attributeId', 'imageId'))
    )

    # Normalize all values, since image sizes vary
    normalizedConfusionMatrics = confusionMatrics.apply(
        normalizeConfusionMatrix,
        axis='columns'
    )

    scores: Dict[str, Dict[str, float]] = {}
    for attribute in sorted(confusionMatrics.index.unique('attributeId')):
        attributeConfusionMatrics = normalizedConfusionMatrics.loc(axis=0)[attribute, :]
        sumAttributeConfusionMatrics = attributeConfusionMatrics.sum(axis='index')

        scores[attribute] = {
            'jaccard': metrics.binaryJaccard(sumAttributeConfusionMatrics),
            'dice': metrics.binaryDice(sumAttributeConfusionMatrics)
        }

    sumConfusionMatrix = normalizedConfusionMatrics.sum(axis='index')
    scores['micro_average'] = {
        'jaccard': metrics.binaryJaccard(sumConfusionMatrix),
        'dice': metrics.binaryDice(sumConfusionMatrix)
    }

    return scores
