import io

import pandas as pd
import pytest

from isic_challenge_scoring import load_csv
from isic_challenge_scoring.types import ScoreError


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


def test_parse_csv_no_newlines(categories):
    prediction_file_stream = io.StringIO('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
    for i in range(10000):
        # Simulate many long floats
        prediction_file_stream.write(f'{i:030f},')
    prediction_file_stream.seek(0)

    with pytest.raises(ScoreError, match=r'^No newlines detected in CSV\.$'):
        load_csv.parse_csv(prediction_file_stream, categories)


def test_parse_csv_empty(categories):
    # Provide just enough to evade the newline check, but raise an EmptyDataError
    prediction_file_stream = io.StringIO('\n\n')

    with pytest.raises(
        ScoreError, match=r'^Could not parse CSV: "No columns to parse from file"\.$'
    ):
        load_csv.parse_csv(prediction_file_stream, categories)


def test_parse_csv_invalid_unicode(categories):
    prediction_file_stream = io.TextIOWrapper(io.BytesIO(b'\xef'))

    with pytest.raises(
        ScoreError, match=r'^Could not parse CSV: could not decode file as UTF-8\.$'
    ):
        load_csv.parse_csv(prediction_file_stream, categories)


@pytest.mark.filterwarnings(
    'ignore:Length of header or names does not match length of data:pandas.errors.ParserWarning'
)
@pytest.mark.filterwarnings('error')
def test_parse_csv_mismatched_headers(categories):
    prediction_file_stream = io.StringIO(
        'image\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n'
    )

    # Pandas should drop extra columns without headers, but this is a common invalid case
    with pytest.raises(ScoreError, match=r'^Missing columns in CSV:'):
        load_csv.parse_csv(prediction_file_stream, categories)


def test_parse_csv_missing_columns(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,BCC,AKIEC,BKL,DF\n' 'ISIC_0000123,1.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreError, match=r"^Missing columns in CSV: \['NV', 'VASC'\]\.$"):
        load_csv.parse_csv(prediction_file_stream, categories)


def test_parse_csv_extra_columns(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,FOO,NV,BCC,AKIEC,BKL,DF,BAZ,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreError, match=r"^Extra columns in CSV: \['BAZ', 'FOO'\]\.$"):
        load_csv.parse_csv(prediction_file_stream, categories)


def test_parse_csv_misnamed_columns(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,FOO,BCC,AKIEC,BKL,BAZ,VASC\n' 'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(ScoreError, match=r"^Missing columns in CSV: \['DF', 'NV'\]\.$"):
        load_csv.parse_csv(prediction_file_stream, categories)


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

    with pytest.raises(ScoreError, match=r"^Missing column in CSV: 'image'\.$"):
        load_csv.parse_csv(prediction_file_stream, categories)


def test_parse_csv_invalid_type_index(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n' '5,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
    )

    prediction_probabilities = load_csv.parse_csv(prediction_file_stream, categories)

    # Apparent numeric 'image' fields should be coerced to string / NumPy 'O'
    assert pd.api.types.is_object_dtype(prediction_probabilities.index)


def test_parse_csv_missing_values(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,1.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(
        ScoreError,
        match=r"^Missing value\(s\) in CSV for images: \['ISIC_0000124', 'ISIC_0000125'\]\.$",
    ):
        load_csv.parse_csv(prediction_file_stream, categories)


def test_parse_csv_non_float_columns(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        "ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,'BAD'\n"
        'ISIC_0000125,0.0,0.0,True,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(
        ScoreError,
        match=r"^CSV contains non-floating-point value\(s\) in columns: \['BCC', 'VASC'\]\.$",
    ):
        load_csv.parse_csv(prediction_file_stream, categories)


def test_parse_csv_out_of_range_values(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,100.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000125,0.0,0.0,-1.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(
        ScoreError,
        match=r'^Values in CSV are outside the interval \[0\.0, 1\.0\] for images: '
        r"\['ISIC_0000123', 'ISIC_0000125'\]\.$",
    ):
        load_csv.parse_csv(prediction_file_stream, categories)


def test_parse_csv_duplicate_images(categories):
    prediction_file_stream = io.StringIO(
        'image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n'
        'ISIC_0000123,1.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000123,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n'
        'ISIC_0000124.jpg,0.0,0.0,1.0,0.0,0.0,0.0,0.0\n'
    )

    with pytest.raises(
        ScoreError,
        match=r"^Duplicate image rows detected in CSV: \['ISIC_0000123', 'ISIC_0000124'\]\.$",
    ):
        load_csv.parse_csv(prediction_file_stream, categories)


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
    with pytest.raises(ScoreError, match=r"^Missing images in CSV: \['ISIC_0000124'\]\.$"):
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)

    truth_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        index=['ISIC_0000123', 'ISIC_0000124'],
        columns=categories,
    )
    prediction_probabilities = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=['ISIC_0000120'], columns=categories
    )
    with pytest.raises(
        ScoreError, match=r"^Missing images in CSV: \['ISIC_0000123', 'ISIC_0000124'\]\.$"
    ):
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)

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
    with pytest.raises(ScoreError, match=r"^Missing images in CSV: \['ISIC_0000124'\]\.$"):
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)


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
    with pytest.raises(
        ScoreError, match=r"^Extra images in CSV: \['ISIC_0000126', 'ISIC_0000127'\]\.$"
    ):
        load_csv.validate_rows(truth_probabilities, prediction_probabilities)


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
