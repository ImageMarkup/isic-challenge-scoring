# -*- coding: utf-8 -*-
import pathlib
import sys

import click
import click_pathlib

from isic_challenge_scoring import score_all
from isic_challenge_scoring.types import ScoreException


DirectoryPath = click_pathlib.Path(exists=True, file_okay=False, dir_okay=True, readable=True)


@click.command(name='isic-challenge-scoring', help='ISIC Challenge submission scoring')
@click.option(
    'truth_input_path',
    '--groundtruth',
    required=True,
    type=DirectoryPath,
    help='path to the ground truth directory',
)
@click.option(
    'prediction_input_path',
    '--submission',
    required=True,
    type=DirectoryPath,
    help='path to the submission directory',
)
@click.option(
    'task_num',
    '--task',
    required=True,
    type=click.IntRange(1, 3),
    help='challenge task number (1, 2, or 3)',
)
@click.option('--require-manuscript', 'require_manuscript', is_flag=True, default=False)
def main(
    truth_input_path: pathlib.Path,
    prediction_input_path: pathlib.Path,
    task_num: int,
    require_manuscript: bool,
):
    try:
        score_all(truth_input_path, prediction_input_path, task_num, require_manuscript)
    except ScoreException as e:
        covalic_error_prefix = 'covalic.error: '
        print(covalic_error_prefix + str(e), file=sys.stderr)
        sys.exit(1)


main()
