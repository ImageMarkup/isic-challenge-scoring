from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import cast, Dict, TextIO

import pandas as pd

from isic_challenge_scoring import metrics
from isic_challenge_scoring.confusion import create_binary_confusion_matrix
from isic_challenge_scoring.load_csv import parse_csv, parse_truth_csv, sort_rows, validate_rows
from isic_challenge_scoring.types import DataFrameDict, RocDict, Score, ScoreDict, SeriesDict


@dataclass(init=False)
class ClassificationScore(Score):
    per_category: pd.DataFrame
    macro_average: pd.Series
    rocs: Dict[str, pd.DataFrame]
    aggregate: pd.Series

    def __init__(
        self,
        truth_probabilities: pd.DataFrame,
        prediction_probabilities: pd.DataFrame,
        truth_weights: pd.DataFrame,
    ) -> None:
        categories = truth_probabilities.columns

        self.per_category = pd.DataFrame(
            [
                self._category_score(
                    truth_probabilities[category],
                    prediction_probabilities[category],
                    truth_weights,
                    category,
                )
                for category in categories
            ]
        )
        self.macro_average = self.per_category.mean(axis='index').rename(
            'macro_average', inplace=True
        )
        self.rocs = {
            category: metrics.roc(
                truth_probabilities[category],
                prediction_probabilities[category],
                truth_weights.score_weight,
            )
            for category in categories
        }
        # Multi-category aggregate metrics
        self.aggregate = pd.Series(
            {
                'balanced_accuracy': metrics.balanced_multiclass_accuracy(
                    truth_probabilities, prediction_probabilities, truth_weights.score_weight
                )
            },
            index=['balanced_accuracy'],
            name='aggregate',
        )

        self.overall = self.aggregate.at['balanced_accuracy']
        self.validation = metrics.balanced_multiclass_accuracy(
            truth_probabilities, prediction_probabilities, truth_weights.validation_weight
        )

    @staticmethod
    def _category_score(
        truth_category_probabilities: pd.Series,
        prediction_category_probabilities: pd.Series,
        truth_weights: pd.DataFrame,
        category: str,
    ) -> pd.Series:
        truth_binary_values: pd.Series = truth_category_probabilities.gt(0.5)
        prediction_binary_values: pd.Series = prediction_category_probabilities.gt(0.5)

        category_cm = create_binary_confusion_matrix(
            truth_binary_values=truth_binary_values.to_numpy(),
            prediction_binary_values=prediction_binary_values.to_numpy(),
            weights=truth_weights.score_weight.to_numpy(),
            name=category,
        )

        return pd.Series(
            {
                'accuracy': metrics.binary_accuracy(category_cm),
                'sensitivity': metrics.binary_sensitivity(category_cm),
                'specificity': metrics.binary_specificity(category_cm),
                'dice': metrics.binary_dice(category_cm),
                'ppv': metrics.binary_ppv(category_cm),
                'npv': metrics.binary_npv(category_cm),
                'auc': metrics.auc(
                    truth_category_probabilities,
                    prediction_category_probabilities,
                    truth_weights.score_weight,
                ),
                'auc_sens_80': metrics.auc_above_sensitivity(
                    truth_category_probabilities,
                    prediction_category_probabilities,
                    truth_weights.score_weight,
                    0.80,
                ),
                'ap': metrics.average_precision(
                    truth_category_probabilities,
                    prediction_category_probabilities,
                    truth_weights.score_weight,
                ),
            },
            index=[
                'accuracy',
                'sensitivity',
                'specificity',
                'dice',
                'ppv',
                'npv',
                'auc',
                'auc_sens_80',
                'ap',
            ],
            name=category,
        )

    def to_string(self) -> str:
        output = super().to_string()
        output += '\n\nPer-category metrics:\n'
        output += self.per_category.to_string()
        output += '\n\nMacro averaged metrics:\n'
        output += self.macro_average.to_string()
        output += '\n\nAggregate metrics:\n'
        output += self.aggregate.to_string()
        return output

    def to_dict(self, rocs: bool = True) -> ScoreDict:
        output = super().to_dict()
        output.update(
            {
                'per_category': cast(DataFrameDict, self.per_category.to_dict()),
                'macro_average': cast(SeriesDict, self.macro_average.to_dict()),
                'aggregate': cast(SeriesDict, self.aggregate.to_dict()),
            }
        )
        if rocs:
            output['rocs'] = {
                category: cast(
                    RocDict,
                    # orient='list' uses ~68% as much space to JSON serialize than orient='records'
                    roc.reset_index().rename(columns={'index': 'threshold'}).to_dict(orient='list'),
                )
                for category, roc in self.rocs.items()
            }
        return output

    @classmethod
    def from_stream(
        cls, truth_file_stream: TextIO, prediction_file_stream: TextIO
    ) -> ClassificationScore:
        truth_probabilities, truth_weights = parse_truth_csv(truth_file_stream)
        categories = truth_probabilities.columns
        prediction_probabilities = parse_csv(prediction_file_stream, categories)

        validate_rows(truth_probabilities, prediction_probabilities)

        sort_rows(truth_probabilities)
        sort_rows(prediction_probabilities)

        score = cls(truth_probabilities, prediction_probabilities, truth_weights)
        return score

    @classmethod
    def from_file(
        cls, truth_file: pathlib.Path, prediction_file: pathlib.Path
    ) -> ClassificationScore:
        with truth_file.open('r') as truth_file_stream, prediction_file.open(
            'r'
        ) as prediction_file_stream:
            return cls.from_stream(
                cast(TextIO, truth_file_stream), cast(TextIO, prediction_file_stream)
            )
