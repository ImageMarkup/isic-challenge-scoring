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

import argparse
import pathlib
import sys

from . import scoreAll
from .scoreCommon import ScoreException


def main():
    parser = argparse.ArgumentParser(
        description='Submission scoring helper script')
    parser.add_argument('-g', '--groundtruth', required=True,
                        help='path to the ground truth directory')
    parser.add_argument('-s', '--submission', required=True,
                        help='path to the submission directory')
    args = parser.parse_args()

    try:
        truthInputPath = pathlib.Path(args.groundtruth)
        if not truthInputPath.is_dir():
            raise ScoreException(
                'Internal error: "--groundtruth" argument must reference a directory')

        predictionInputPath = pathlib.Path(args.submission)
        if not predictionInputPath.is_dir():
            raise ScoreException(
                'Internal error: "--submission" argument must reference a directory')

        scoreAll(truthInputPath, predictionInputPath)
    except ScoreException as e:
        covalicErrorPrefix = 'covalic.error: '
        print(covalicErrorPrefix + str(e), file=sys.stderr)
        exit(1)


main()
