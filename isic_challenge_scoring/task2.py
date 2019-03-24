# -*- coding: utf-8 -*-
import pathlib
from typing import Dict, List

import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import createBinaryConfusionMatrix, normalizeConfusionMatrix
from isic_challenge_scoring.load_image import iterImagePairs


def score(truthPath: pathlib.Path, predictionPath: pathlib.Path) -> List[Dict]:
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

    scores = []
    for attribute in sorted(confusionMatrics.index.unique('attributeId')):
        attributeConfusionMatrics = normalizedConfusionMatrics.loc(axis=0)[attribute, :]
        sumAttributeConfusionMatrics = attributeConfusionMatrics.sum(axis='index')
        scores.append({
            'dataset': attribute,
            'metrics': [
                {
                    'name': 'jaccard',
                    'value': metrics.binaryJaccard(sumAttributeConfusionMatrics)
                },
                {
                    'name': 'dice',
                    'value': metrics.binaryDice(sumAttributeConfusionMatrics)
                },
            ]
        })

    sumConfusionMatrix = normalizedConfusionMatrics.sum(axis='index')
    scores.extend([
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
    ])

    return scores
