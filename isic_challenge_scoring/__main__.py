import json
import pathlib
from typing import cast

import click
import click_pathlib

from isic_challenge_scoring.classification import ClassificationScore
from isic_challenge_scoring.segmentation import SegmentationScore
from isic_challenge_scoring.types import ScoreException


DirectoryPath = click_pathlib.Path(exists=True, file_okay=False, dir_okay=True, readable=True)
FilePath = click_pathlib.Path(exists=True, file_okay=True, dir_okay=False, readable=True)


@click.group(name='isic-challenge-scoring', help='ISIC Challenge submission scoring')
@click.option('-o', '--output', type=click.Choice(['table', 'json']), default='table')
def cli(output: str) -> None:
    pass


@cli.command()
@click.pass_context
@click.argument('truth_dir', type=DirectoryPath)
@click.argument('prediction_dir', type=DirectoryPath)
def segmentation(ctx: click.Context, truth_dir: pathlib.Path, prediction_dir: pathlib.Path) -> None:
    try:
        score = SegmentationScore.from_dir(truth_dir, prediction_dir)
    except ScoreException as e:
        raise click.ClickException(str(e))

    output: str = cast(click.Context, ctx.parent).params['output']
    if output == 'table':
        click.echo(score.to_string())
    elif output == 'json':
        click.echo(json.dumps(score.to_dict(), indent=2))


@cli.command()
@click.pass_context
@click.argument('truth_file', type=FilePath)
@click.argument('prediction_file', type=FilePath)
def classification(
    ctx: click.Context, truth_file: pathlib.Path, prediction_file: pathlib.Path
) -> None:
    try:
        score = ClassificationScore.from_file(truth_file, prediction_file)
    except ScoreException as e:
        raise click.ClickException(str(e))

    output: str = cast(click.Context, ctx.parent).params['output']
    if output == 'table':
        click.echo(score.to_string())
    elif output == 'json':
        click.echo(json.dumps(score.to_dict(rocs=False), indent=2))


if __name__ == '__main__':
    cli()
