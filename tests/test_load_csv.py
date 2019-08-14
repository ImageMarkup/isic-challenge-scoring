# -*- coding: utf-8 -*-
import io

import pandas as pd
import pytest

from isic_challenge_scoring import load_csv
from isic_challenge_scoring.types import ScoreException


def test_parse_truth_csv(categories):
    truth_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC,score_weight,validation_weight\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0\n'
        'ISIC_0000125,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    truth_probabilities, truth_weights = load_csv.parse_truth_csv(truth_file_stream)

    assert truth_probabilities.equals(
        pd.DataFrame(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=['ISIC_0000123', 'ISIC_0000124', 'ISIC_0000125'],
            columns=categories,
        )
    )
    assert truth_weights.equals(
        pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            index=['ISIC_0000123', 'ISIC_0000124', 'ISIC_0000125'],
            columns=pd.Index(['score_weight', 'validation_weight']),
        )
    )


def test_parse_truth_csv_legacy(categories):
    truth_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n'
    )

    truth_probabilities, truth_weights = load_csv.parse_truth_csv(truth_file_stream)

    assert truth_probabilities.equals(
        pd.DataFrame(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=['ISIC_0000123', 'ISIC_0000124', 'ISIC_0000125'],
            columns=categories,
        )
    )
    assert truth_weights.equals(
        pd.DataFrame(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            index=['ISIC_0000123', 'ISIC_0000124', 'ISIC_0000125'],
            columns=pd.Index(['score_weight', 'validation_weight']),
        )
    )


def test_parse_csv(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124.jpg,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125.JPG,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n'
    )

    prediction_probabilities = load_csv.parse_csv(prediction_file_stream, categories)

    assert prediction_probabilities.equals(
        pd.DataFrame(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=['ISIC_0000123', 'ISIC_0000124', 'ISIC_0000125'],
            columns=categories,
        )
    )


def test_parse_csv_missing_columns(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,BCC,AKIEC,BKL,DF\n' 'ISIC_0000123,1.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, categories)

    assert 'Missing columns in CSV: [\'NV\', \'VASC\'].' == str(exc_info.value)


def test_parse_csv_extra_columns(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,FOO,NV,BCC,AKIEC,BKL,DF,BAZ,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, categories)

    assert 'Extra columns in CSV: [\'BAZ\', \'FOO\'].' == str(exc_info.value)


def test_parse_csv_misnamed_columns(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,FOO,BCC,AKIEC,BKL,BAZ,VASC\n' 'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, categories)

    assert 'Missing columns in CSV: [\'DF\', \'NV\'].' == str(exc_info.value)


def test_parse_csv_trailing_delimiters(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0,\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,0.0,\n'
    )

    # If all data rows have trailing delimiters, 'pd.read_csv' can misinterpret the data without
    # 'index_col=False'
    prediction_probabilities = load_csv.parse_csv(prediction_file_stream, categories)

    assert prediction_probabilities.equals(
        pd.DataFrame(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            index=['ISIC_0000123', 'ISIC_0000124'],
            columns=categories,
        )
    )


def test_parse_csv_reordered_columns(categories):
    prediction_file_stream = io.StringIO(
        'NV,BCC,BKL,DF,AKIEC,MEL,VASC,image\n' '0.0,0.0,0.0,0.0,0.0,1.0,0.0,ISIC_0000123\n'
    )

    prediction_probabilities = load_csv.parse_csv(prediction_file_stream, categories)

    assert prediction_probabilities.equals(
        pd.DataFrame(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=['ISIC_0000123'], columns=categories
        )
    )


def test_parse_csv_missing_index(categories):
    prediction_file_stream = io.StringIO(
        'MEL,NV,BCC,AKIEC,BKL,DF,VASC\n' '1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, categories)

    assert 'Missing column in CSV: "image".' == str(exc_info.value)


def test_parse_csv_invalid_type_index(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n' '5,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    prediction_probabilities = load_csv.parse_csv(prediction_file_stream, categories)

    # Apparent numeric 'image' fields should be coerced to string / NumPy 'O'
    assert prediction_probabilities.index.is_object()


def test_parse_csv_missing_values(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,1.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, categories)

    assert 'Missing value(s) in CSV for images: [\'ISIC_0000124\', \'ISIC_0000125\'].' == str(
        exc_info.value
    )


def test_parse_csv_non_float_columns(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,\'BAD\'\n'
        'ISIC_0000125,0.0,0.0,True,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, categories)

    assert 'CSV contains non-floating-point value(s) in columns: [\'BCC\', \'VASC\'].' == str(
        exc_info.value
    )


def test_parse_csv_out_of_range_values(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,100.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,-1.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, categories)

    assert (
        'Values in CSV are outside the interval [0.0, 1.0] for images: '
        '[\'ISIC_0000123\', \'ISIC_0000125\'].' == str(exc_info.value)
    )


def test_parse_csv_duplicate_images(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000123,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124.jpg,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreException) as exc_info:
        load_csv.parse_csv(prediction_file_stream, categories)

    assert 'Duplicate image rows detected in CSV: [\'ISIC_0000123\', \'ISIC_0000124\'].' == str(
        exc_info.value
    )


def test_validate_rows_missing_images(categories):
    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=categories,
    )
    prediction_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=categories,
    )
    load_csv.validate_rows(truth_probabilities, prediction_probabilities)

    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=categories,
    )
    prediction_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=['ISIC_0000123'], columns=categories
    )
    with pytest.raises(ScoreException) as exc_info:
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)
    assert 'Missing images in CSV: [\'ISIC_0000124\'].' == str(exc_info.value)

    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=categories,
    )
    prediction_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=['ISIC_0000120'], columns=categories
    )
    with pytest.raises(ScoreException) as exc_info:
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)
    assert 'Missing images in CSV: [\'ISIC_0000123\', \'ISIC_0000124\'].' == str(exc_info.value)

    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=categories,
    )
    prediction_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000125'],
        columns=categories,
    )
    with pytest.raises(ScoreException) as exc_info:
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)
    assert 'Missing images in CSV: [\'ISIC_0000124\'].' == str(exc_info.value)


def test_validate_rows_extra_images(categories):
    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=['ISIC_0000123'], columns=categories
    )
    prediction_probabilities = pd.DataFrame(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        index=['ISIC_0000123', 'ISIC_0000126', 'ISIC_0000127'],
        columns=categories,
    )
    with pytest.raises(ScoreException) as exc_info:
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)
    assert 'Extra images in CSV: [\'ISIC_0000126\', \'ISIC_0000127\'].' == str(exc_info.value)


def test_sort_rows(categories):
    prediction_probabilities = pd.DataFrame(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        index=['ISIC_0000124', 'ISIC_0000125', 'ISIC_0000123'],
        columns=categories,
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
            columns=categories,
        )
    )
