# -*- coding: utf-8 -*-
import json
import os
import pathlib
import shutil
import tempfile
from typing import Tuple
import zipfile

from isic_challenge_scoring import task1
from isic_challenge_scoring import task2
from isic_challenge_scoring import task3
from isic_challenge_scoring.types import ScoreException, ScoresType


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


def unzip_all(input_path: pathlib.Path) -> Tuple[pathlib.Path, tempfile.TemporaryDirectory]:
    """
    Extract / copy all files in directory.

    Validates that the path contains exactly one file. Return a path to the extracted content.
    """
    input_files = [f for f in input_path.iterdir() if f.is_file()]
    input_dirs = [f for f in input_path.iterdir() if f.is_dir()]

    if len(input_files) > 1:
        raise Exception('Multiple files submitted. Exactly one ZIP file should be submitted.')
    elif len(input_files) < 1:
        raise Exception('No files submitted. Exactly one ZIP file should be submitted.')

    input_file = input_files[0]

    if input_dirs:
        # Expect only files
        raise Exception(f'Unexpected directories found: {sorted(map(str, input_dirs))}')

    output_temp_dir = tempfile.TemporaryDirectory()
    output_path = pathlib.Path(output_temp_dir.name)

    if input_file.suffix.lower() == '.zip':
        extract_zip(input_file, output_path)
    else:
        shutil.copy(str(input_file), str(output_path))

    return output_path, output_temp_dir


def score_all(truth_input_path: pathlib.Path, prediction_input_path: pathlib.Path, task_num: int):
    # Unzip zip files contained in the input folders
    truth_path, truth_temp_dir = unzip_all(truth_input_path)

    prediction_path, prediction_temp_dir = unzip_all(prediction_input_path)

    if task_num == 1:
        score = task1.score
    elif task_num == 2:
        score = task2.score
    elif task_num == 3:
        score = task3.score
    else:
        raise Exception(f'Unknown ground truth phase number: {task_num}.')
    scores: ScoresType = score(truth_path, prediction_path)

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
