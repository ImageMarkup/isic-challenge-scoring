from importlib.metadata import PackageNotFoundError, version

from isic_challenge_scoring.classification import ClassificationMetric, ClassificationScore
from isic_challenge_scoring.segmentation import SegmentationScore
from isic_challenge_scoring.types import ScoreError

ScoreException = ScoreError

__all__ = ['ClassificationScore', 'SegmentationScore', 'ScoreError', 'ClassificationMetric']

try:
    __version__ = version('isic-challenge-scoring')
except PackageNotFoundError:
    # package is not installed
    pass
