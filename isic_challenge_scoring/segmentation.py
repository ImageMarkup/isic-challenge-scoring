from __future__ import annotations

from dataclasses import dataclass
import enum
import pathlib
from typing import Iterable, cast

import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import create_binary_confusion_matrix
from isic_challenge_scoring.load_image import ImagePair, iter_image_pairs
from isic_challenge_scoring.types import Score, ScoreDict, SeriesDict
from isic_challenge_scoring.unzip import unzip_all


class SegmentationMetric(enum.Enum):
    JACCARD = 'jaccard'  # 2016/1  2016/2b  2017/1  2018/2
    THRESHOLD_JACCARD = 'threshold_jaccard'  # 2018/1
    AUC = 'auc'  # 2017/2


@dataclass(init=False)
class SegmentationScore(Score):
    macro_average: pd.Series

    def __init__(self, image_pairs: Iterable[ImagePair], target_metric: SegmentationMetric) -> None:
        # TODO: Add weighting
        confusion_matrics = pd.DataFrame(
            [
                create_binary_confusion_matrix(
                    truth_binary_values=image_pair.truth_image > 128,
                    prediction_binary_values=image_pair.prediction_image > 128,
                    name=image_pair.image_id,
                )
                for image_pair in image_pairs
            ]
        )

        per_image = pd.DataFrame(
            {
                'accuracy': confusion_matrics.apply(metrics.binary_accuracy, axis='columns'),
                'sensitivity': confusion_matrics.apply(metrics.binary_sensitivity, axis='columns'),
                'specificity': confusion_matrics.apply(metrics.binary_specificity, axis='columns'),
                'jaccard': confusion_matrics.apply(metrics.binary_jaccard, axis='columns'),
                'threshold_jaccard': confusion_matrics.apply(
                    metrics.binary_threshold_jaccard, threshold=0.65, axis='columns'
                ),
                'dice': confusion_matrics.apply(metrics.binary_dice, axis='columns'),
            },
            columns=[
                'accuracy',
                'sensitivity',
                'specificity',
                'jaccard',
                'threshold_jaccard',
                'dice',
            ],
        )

        self.macro_average = per_image.mean(axis='index').rename('macro_average', inplace=True)

        if target_metric == SegmentationMetric.JACCARD:
            self.overall = self.macro_average.at['jaccard']
            self.validation = self.macro_average.at['jaccard']
        elif target_metric == SegmentationMetric.THRESHOLD_JACCARD:
            self.overall = self.macro_average.at['threshold_jaccard']
            self.validation = self.macro_average.at['threshold_jaccard']
        elif target_metric == SegmentationMetric.AUC:
            raise Exception

    def to_string(self) -> str:
        output = super().to_string()
        output += '\n\nMacro averaged metrics:\n'
        output += self.macro_average.to_string()
        return output

    def to_dict(self) -> ScoreDict:
        output = super().to_dict()
        output.update({'macro_average': cast(SeriesDict, self.macro_average.to_dict())})
        return output

    @classmethod
    def from_dir(
        cls,
        truth_path: pathlib.Path,
        prediction_path: pathlib.Path,
        target_metric: SegmentationMetric,
    ) -> SegmentationScore:
        image_pairs = iter_image_pairs(truth_path, prediction_path)
        return cls(image_pairs, target_metric)

    @classmethod
    def from_zip_file(
        cls,
        truth_zip_file: pathlib.Path,
        prediction_zip_file: pathlib.Path,
        target_metric: SegmentationMetric,
    ) -> SegmentationScore:
        truth_path, truth_temp_dir = unzip_all(truth_zip_file)
        # TODO: If an exception occurs while unzipping prediction_zip_file, truth_temp_dir is not
        #  cleaned up
        prediction_path, prediction_temp_dir = unzip_all(prediction_zip_file)

        try:
            score = cls.from_dir(truth_path, prediction_path, target_metric)
        finally:
            truth_temp_dir.cleanup()
            prediction_temp_dir.cleanup()

        return score
