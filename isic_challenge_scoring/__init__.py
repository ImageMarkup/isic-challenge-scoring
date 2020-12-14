from importlib.metadata import PackageNotFoundError, version

from isic_challenge_scoring.classification import ClassificationScore, ValidationMetric
from isic_challenge_scoring.segmentation import SegmentationScore
from isic_challenge_scoring.types import ScoreException

__all__ = ['ClassificationScore', 'SegmentationScore', 'ScoreException', 'ValidationMetric']

try:
    __version__ = version('isic-challenge-scoring')
except PackageNotFoundError:
    # package is not installed
    pass
