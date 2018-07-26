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

import json
import os
import pathlib
import re
import shutil
import tempfile
import zipfile

from .scoreCommon import ScoreException
from .task1 import scoreP1
from .task2 import scoreP2
from .task3 import scoreP3


def extractZip(zipPath, outputPath, flatten=True):
    """
    Extract a zip file, optionally flattening it into a single directory.
    """
    try:
        with zipfile.ZipFile(zipPath) as zf:
            if flatten:
                for memberInfo in zf.infolist():
                    memberName = memberInfo.filename
                    if memberName.startswith('__MACOSX'):
                        # Ignore Mac OS X metadata
                        continue

                    memberBaseName = os.path.basename(memberName)
                    if not memberBaseName:
                        # Skip directories
                        continue

                    memberOutputPath = outputPath / memberBaseName

                    with zf.open(memberInfo) as inputStream, \
                            memberOutputPath.open('wb') as outputStream:
                        shutil.copyfileobj(inputStream, outputStream)
            else:
                zf.extractall(outputPath)
    except zipfile.BadZipfile as e:
        raise ScoreException(f'Could not read ZIP file "{zipPath.name}": {str(e)}.')


def unzipAll(inputPath):
    """
    Extract / copy all files in directory.
    Return a path to the extracted content.
    """
    inputFiles = list(inputPath.iterdir())
    if len(inputFiles) > 1:
        raise ScoreException(
            'Multiple files submitted. Exactly one ZIP or CSV file should be submitted.')
    elif len(inputFiles) < 1:
        raise ScoreException(
            'No files submitted. Exactly one ZIP or CSV file should be submitted.')

    inputFile = inputFiles[0]
    if not inputFile.is_file():
        # Covalic should not allow this to happen
        raise ScoreException('Internal error: non-regular file submitted.')

    outputTempDir = tempfile.TemporaryDirectory()
    outputPath = pathlib.Path(outputTempDir.name)

    if inputFile.suffix.lower() == '.zip':
        extractZip(inputFile, outputPath)
    else:
        shutil.copy(inputFile, outputPath)

    return outputPath, outputTempDir


def scoreAll(truthInputPath, predictionInputPath):
    # Unzip zip files contained in the input folders
    truthPath, truthTempDir = unzipAll(truthInputPath)

    predictionPath, predictionTempDir = unzipAll(predictionInputPath)

    # Identify which phase this is, based on ground truth file name
    truthRe = re.match(
        r'^ISIC2018_Task(?P<taskNum>[0-9])_(?P<phaseType>Validation|Test)_GroundTruth\.zip$',
        next(truthInputPath.iterdir()).name)
    if not truthRe:
        raise ScoreException(
            f'Internal error: could not parse ground truth file name: {truthInputPath.name}.')

    taskNum = truthRe.group('taskNum')
    if taskNum == '1':
        scores = scoreP1(truthPath, predictionPath)
    elif taskNum == '2':
        scores = scoreP2(truthPath, predictionPath)
    elif taskNum == '3':
        scores = scoreP3(truthPath, predictionPath)
    else:
        raise ScoreException(
            f'Internal error: unknown ground truth phase number: {truthInputPath.name}.')

    print(json.dumps(scores))

    truthTempDir.cleanup()
    predictionTempDir.cleanup()
