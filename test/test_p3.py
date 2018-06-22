import io

import pandas as pd
import pytest

from isic_challenge_scoring import task3
from isic_challenge_scoring.scoreCommon import ScoreException


def test_parseCsv():
    predictionFileStream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n'
    )

    predictionProbabilities = task3.parseCsv(predictionFileStream)

    assert predictionProbabilities.equals(pd.DataFrame([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ], index=['ISIC_0000123', 'ISIC_0000124', 'ISIC_0000125'], columns=task3.CATEGORIES))


def test_parseCsv_missingColumns():
    predictionFileStream = io.StringIO(
        'image,MEL,BCC,AKIEC,BKL,DF\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as excInfo:
        task3.parseCsv(predictionFileStream)

    assert 'Missing columns in CSV: [\'NV\', \'VASC\']' == str(excInfo.value)


def test_parseCsv_extraColumns():
    predictionFileStream = io.StringIO(
        'image,MEL,FOO,NV,BCC,AKIEC,BKL,DF,BAZ,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as excInfo:
        task3.parseCsv(predictionFileStream)

    assert 'Extra columns in CSV: [\'BAZ\', \'FOO\']' == str(excInfo.value)


def test_parseCsv_misnamedColumns():
    predictionFileStream = io.StringIO(
        'image,MEL,FOO,BCC,AKIEC,BKL,BAZ,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as excInfo:
        task3.parseCsv(predictionFileStream)

    assert 'Missing columns in CSV: [\'DF\', \'NV\']' == str(excInfo.value)


def test_parseCsv_reorderedColumns():
    predictionFileStream = io.StringIO(
        'NV,BCC,BKL,DF,AKIEC,MEL,VASC,image\n'
        '0.0,0.0,0.0,0.0,0.0,1.0,0.0,ISIC_0000123\n'
    )

    predictionProbabilities = task3.parseCsv(predictionFileStream)

    assert predictionProbabilities.equals(pd.DataFrame([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], index=['ISIC_0000123'], columns=task3.CATEGORIES))


def test_parseCsv_missingIndex():
    predictionFileStream = io.StringIO(
        'MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        '1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as excInfo:
        task3.parseCsv(predictionFileStream)

    assert 'Missing column in CSV: "image"' == str(excInfo.value)


def test_toLabels():
    probabilities = pd.DataFrame([
        # NV
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        # undecided
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # AKIEC
        [0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2],
        # undecided
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        # MEL
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], columns=task3.CATEGORIES)

    labels = task3.toLabels(probabilities)

    assert labels.equals(pd.Series([
        'NV',
        'undecided',
        'AKIEC',
        'undecided',
        'MEL'
    ]))


def test_getFrequencies():
    labels = pd.Series(['MEL', 'MEL', 'VASC', 'AKIEC'])

    labelFrequencies = task3.getFrequencies(labels)

    assert labelFrequencies.equals(pd.Series({
        'MEL': 2,
        'NV': 0,
        'BCC': 0,
        'AKIEC': 1,
        'BKL': 0,
        'DF': 0,
        'VASC': 1
    }))
    # Ensure the ordering is correct (although Python3.6 dicts are ordered)
    assert labelFrequencies.index.equals(task3.CATEGORIES)
