# -*- coding: utf-8 -*-
import json
import os
import pathlib
import shutil
import tempfile
from typing import Tuple
import zipfile

from isic_challenge_scoring.exception import ScoreException
from isic_challenge_scoring.task1 import score as score_task1
from isic_challenge_scoring.task2 import score as score_task2
from isic_challenge_scoring.task3 import score as score_task3


def extract_zip(zip_path: pathlib.Path, output_path: pathlib.Path, flatten: bool = True):
    """Extract a zip file, optionally flattening it into a single directory."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if flatten:
                for member_info in zf.infolist():
                    member_name = member_info.filename
                    if member_name.startswith('__MACOSX'):
                        # Ignore Mac OS X metadata
                        continue

                    member_base_name = os.path.basename(member_name)
                    if not member_base_name:
                        # Skip directories
                        continue

                    member_output_path = output_path / member_base_name

                    with zf.open(member_info) as input_stream, member_output_path.open(
                        'wb'
                    ) as output_stream:
                        shutil.copyfileobj(input_stream, output_stream)
            else:
                zf.extractall(output_path)
    except zipfile.BadZipfile as e:
        raise ScoreException(f'Could not read ZIP file "{zip_path.name}": {str(e)}.')


def unzip_all(
    input_path: pathlib.Path, allow_manuscript_directory: bool = False
) -> Tuple[pathlib.Path, tempfile.TemporaryDirectory]:
    """
    Extract / copy all files in directory.

    Validates that the path contains exactly one file. Optionally allow an 'Abstract' directory to
    exist which contains exactly one manuscript file. Return a path to the extracted content.
    """
    input_files = [f for f in input_path.iterdir() if f.is_file()]
    input_dirs = [f for f in input_path.iterdir() if f.is_dir()]

    if len(input_files) > 1:
        raise ScoreException('Multiple files submitted. Exactly one ZIP file should be submitted.')
    elif len(input_files) < 1:
        raise ScoreException('No files submitted. Exactly one ZIP file should be submitted.')

    input_file = input_files[0]

    manuscript_file = None

    if allow_manuscript_directory:
        if len(input_dirs) > 1:
            raise ScoreException('Internal error: multiple directories found.')
        elif len(input_dirs) == 1:
            input_dir = input_dirs[0]
            if input_dir.name != 'Abstract':
                raise ScoreException(
                    f'Internal error: unexpected directory found: {input_dir.name}.'
                )

            manuscript_files = list(input_dir.iterdir())
            if not manuscript_files:
                raise ScoreException('Empty manuscript directory found.')
            elif len(manuscript_files) > 1:
                raise ScoreException('Multiple files found in manuscript directory.')

            manuscript_file = manuscript_files[0]
    elif input_dirs:
        # Expect only files
        raise ScoreException('Internal error: unexpected directory found.')

    output_temp_dir = tempfile.TemporaryDirectory()
    output_path = pathlib.Path(output_temp_dir.name)

    if input_file.suffix.lower() == '.zip':
        extract_zip(input_file, output_path)
    else:
        shutil.copy(input_file, output_path)

    if manuscript_file is not None:
        shutil.copy(manuscript_file, output_path)

    return output_path, output_temp_dir


def ensure_manuscript(prediction_path: pathlib.Path):
    manuscript_file_count = sum(
        manuscript_file.suffix.lower() == '.pdf' for manuscript_file in prediction_path.iterdir()
    )
    if manuscript_file_count > 1:
        raise ScoreException(
            'Multiple PDFs submitted. Exactly one PDF file, containing the descriptive manuscript, '
            'must included in the submission.'
        )
    elif manuscript_file_count < 1:
        raise ScoreException(
            'No PDF submitted. Exactly one PDF file, containing the descriptive manuscript, '
            'must included in the submission.'
        )


def score_all(
    truth_input_path: pathlib.Path,
    prediction_input_path: pathlib.Path,
    task_num: int,
    require_manuscript: bool,
):
    # Unzip zip files contained in the input folders
    truth_path, truth_temp_dir = unzip_all(truth_input_path)

    prediction_path, prediction_temp_dir = unzip_all(
        prediction_input_path, allow_manuscript_directory=True
    )

    if require_manuscript:
        ensure_manuscript(prediction_path)

    if task_num == 1:
        scores = score_task1(truth_path, prediction_path)
    elif task_num == 2:
        scores = score_task2(truth_path, prediction_path)
    elif task_num == 3:
        scores = score_task3(truth_path, prediction_path)
    else:
        raise ScoreException(f'Internal error: unknown ground truth phase number: {task_num}.')

    # Output in Covalic format
    print(
        json.dumps(
            [
                {
                    'dataset': dataset,
                    'metrics': [
                        {'name': metric_name, 'value': metric_value}
                        for metric_name, metric_value in metrics.items()
                    ],
                }
                for dataset, metrics in scores.items()
            ]
        )
    )

    truth_temp_dir.cleanup()
    prediction_temp_dir.cleanup()
