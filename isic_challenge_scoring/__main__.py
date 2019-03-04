# -*- coding: utf-8 -*-

###############################################################################
#  Copyright Kitware Inc.
#
#  Licensed under the Apache License, Version 2.0 ( the "License" );
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import click
import pathlib
import sys

from . import scoreAll
from .scoreCommon import ScoreException

DirectoryPath = click.Path(exists=True, file_okay=False, dir_okay=True, readable=True)


@click.command(name='isic-challenge-scoring', help='ISIC Challenge submission scoring')
@click.option('truthInputPath', '--groundtruth', required=True, type=DirectoryPath,
              help='path to the ground truth directory')
@click.option('predictionInputPath', '--submission', required=True, type=DirectoryPath,
              help='path to the submission directory')
@click.option('taskNum', '--task', required=True, type=click.IntRange(1, 3),
              help='challenge task number (1, 2, or 3)')
def main(truthInputPath, predictionInputPath, taskNum):
    truthInputPath = pathlib.Path(truthInputPath)
    predictionInputPath = pathlib.Path(predictionInputPath)

    try:
        scoreAll(truthInputPath, predictionInputPath, taskNum)
    except ScoreException as e:
        covalicErrorPrefix = 'covalic.error: '
        print(covalicErrorPrefix + str(e), file=sys.stderr)
        sys.exit(1)


main()
