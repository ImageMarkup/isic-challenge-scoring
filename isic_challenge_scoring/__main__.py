# -*- coding: utf-8 -*-
import json

import click
import click_pathlib

from isic_challenge_scoring import task3
from isic_challenge_scoring.types import ScoreException


DirectoryPath = click_pathlib.Path(exists=True, file_okay=False, dir_okay=True, readable=True)
FilePath = click_pathlib.Path(exists=True, file_okay=True, dir_okay=False, readable=True)


@click.group(name='isic-challenge-scoring', help='ISIC Challenge submission scoring')
def cli():
    pass


@cli.command()
@click.argument('truth_file', type=FilePath)
@click.argument('prediction_file', type=FilePath)
def classification(truth_file, prediction_file):
    try:
        scores = task3.score(truth_file, prediction_file)
    except ScoreException as e:
        raise click.ClickException(str(e))
    else:
        click.echo(json.dumps(scores, indent=2))


if __name__ == '__main__':
    cli()
