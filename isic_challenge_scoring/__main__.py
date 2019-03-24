# -*- coding: utf-8 -*-
import pathlib
import sys

import click

from isic_challenge_scoring import scoreAll
from isic_challenge_scoring.exception import ScoreException


DirectoryPath = click.Path(exists=True, file_okay=False, dir_okay=True, readable=True)


@click.command(name='isic-challenge-scoring', help='ISIC Challenge submission scoring')
@click.option('truthInputPath', '--groundtruth', required=True, type=DirectoryPath,
              help='path to the ground truth directory')
@click.option('predictionInputPath', '--submission', required=True, type=DirectoryPath,
              help='path to the submission directory')
@click.option('taskNum', '--task', required=True, type=click.IntRange(1, 3),
              help='challenge task number (1, 2, or 3)')
@click.option('--require-manuscript', 'requireManuscript', is_flag=True, default=False)
def main(truthInputPath: str, predictionInputPath: str, taskNum: int, requireManuscript: bool):
    truthInputPath = pathlib.Path(truthInputPath)
    predictionInputPath = pathlib.Path(predictionInputPath)

    try:
        scoreAll(truthInputPath, predictionInputPath, taskNum, requireManuscript)
    except ScoreException as e:
        covalicErrorPrefix = 'covalic.error: '
        print(covalicErrorPrefix + str(e), file=sys.stderr)
        sys.exit(1)


main()
