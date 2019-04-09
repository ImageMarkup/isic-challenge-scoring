# -*- coding: utf-8 -*-
import io

import pandas as pd
import pytest

from isic_challenge_scoring import load_csv
from isic_challenge_scoring.exception import ScoreException
from isic_challenge_scoring.task3 import CATEGORIES


def test_parse_csv():
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n'
    )

    prediction_probabilities = load_csv.parse_csv(prediction_file_stream, CATEGORIES)

    assert prediction_probabilities.equals(
        pd.DataFrame(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=['ISIC_0000123', 'ISIC_0000124', 'ISIC_0000125'],
            columns=CATEGORIES,
        )
    )


def test_parse_csv_missing_columns():
    prediction_file_stream = io.StringIO(
        'image,MEL,BCC,AKIEC,BKL,DF\n' 'ISIC_0000123,1.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, CATEGORIES)

    assert 'Missing columns in CSV: [\'NV\', \'VASC\'].' == str(exc_info.value)


def test_parse_csv_extra_columns():
    prediction_file_stream = io.StringIO(
        'image,MEL,FOO,NV,BCC,AKIEC,BKL,DF,BAZ,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, CATEGORIES)

    assert 'Extra columns in CSV: [\'BAZ\', \'FOO\'].' == str(exc_info.value)


def test_parse_csv_misnamed_columns():
    prediction_file_stream = io.StringIO(
        'image,MEL,FOO,BCC,AKIEC,BKL,BAZ,VASC\n' 'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, CATEGORIES)

    assert 'Missing columns in CSV: [\'DF\', \'NV\'].' == str(exc_info.value)


def test_parse_csv_reordered_columns():
    prediction_file_stream = io.StringIO(
        'NV,BCC,BKL,DF,AKIEC,MEL,VASC,image\n' '0.0,0.0,0.0,0.0,0.0,1.0,0.0,ISIC_0000123\n'
    )

    prediction_probabilities = load_csv.parse_csv(prediction_file_stream, CATEGORIES)

    assert prediction_probabilities.equals(
        pd.DataFrame(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=['ISIC_0000123'], columns=CATEGORIES
        )
    )


def test_parse_csv_missing_index():
    prediction_file_stream = io.StringIO(
        'MEL,NV,BCC,AKIEC,BKL,DF,VASC\n' '1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, CATEGORIES)

    assert 'Missing column in CSV: "image".' == str(exc_info.value)


def test_parse_csv_missing_values():
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,1.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, CATEGORIES)

    assert 'Missing value(s) in CSV for images: [\'ISIC_0000124\', \'ISIC_0000125\'].' == str(
        exc_info.value
    )


def test_parse_csv_non_float_columns():
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,\'BAD\'\n'
        'ISIC_0000125,0.0,0.0,True,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, CATEGORIES)

    assert 'CSV contains non-floating-point value(s) in columns: [\'BCC\', \'VASC\'].' == str(
        exc_info.value
    )


def test_parse_csv_out_of_range_values():
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,100.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,-1.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, CATEGORIES)

    assert (
        'Values in CSV are outside the interval [0.0, 1.0] for images: '
        '[\'ISIC_0000123\', \'ISIC_0000125\'].' == str(exc_info.value)
    )


def test_exclude_rows():
    probabilities = pd.DataFrame(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        index=['ISIC_0000123', 'ISIC_0000124', 'ISIC_0000125'],
        columns=CATEGORIES,
    )

    assert probabilities.shape == (3, 7)
    load_csv.exclude_rows(probabilities, ['ISIC_0000123', 'ISIC_0000125'])
    assert probabilities.shape == (1, 7)
    load_csv.exclude_rows(probabilities, ['ISIC_0000123'])
    assert probabilities.shape == (1, 7)


def test_validate_rows_missing_images():
    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=CATEGORIES,
    )
    prediction_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=CATEGORIES,
    )
    load_csv.validate_rows(truth_probabilities, prediction_probabilities)

    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=CATEGORIES,
    )
    prediction_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=['ISIC_0000123'], columns=CATEGORIES
    )
    with pytest.raises(ScoreException) as exc_info:
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)
    assert 'Missing images in CSV: [\'ISIC_0000124\'].' == str(exc_info.value)

    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=CATEGORIES,
    )
    prediction_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=['ISIC_0000120'], columns=CATEGORIES
    )
    with pytest.raises(ScoreException) as exc_info:
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)
    assert 'Missing images in CSV: [\'ISIC_0000123\', \'ISIC_0000124\'].' == str(exc_info.value)

    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=CATEGORIES,
    )
    prediction_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000125'],
        columns=CATEGORIES,
    )
    with pytest.raises(ScoreException) as exc_info:
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)
    assert 'Missing images in CSV: [\'ISIC_0000124\'].' == str(exc_info.value)


def test_validate_rows_extra_images():
    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=['ISIC_0000123'], columns=CATEGORIES
    )
    prediction_probabilities = pd.DataFrame(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        index=['ISIC_0000123', 'ISIC_0000126', 'ISIC_0000127'],
        columns=CATEGORIES,
    )
    with pytest.raises(ScoreException) as exc_info:
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)
    assert 'Extra images in CSV: [\'ISIC_0000126\', \'ISIC_0000127\'].' == str(exc_info.value)


def test_sort_rows():
    prediction_probabilities = pd.DataFrame(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        index=['ISIC_0000124', 'ISIC_0000125', 'ISIC_0000123'],
        columns=CATEGORIES,
    )

    load_csv.sort_rows(prediction_probabilities)

    assert prediction_probabilities.equals(
        pd.DataFrame(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=['ISIC_0000123', 'ISIC_0000124', 'ISIC_0000125'],
            columns=CATEGORIES,
        )
    )
