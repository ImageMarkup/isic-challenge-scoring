#!/usr/bin/env python
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

from __future__ import print_function

import argparse
import json
import os
import re
import sys
import zipfile

from scoreCommon import ScoreException
from scoreP1 import scoreP1
from scoreP2 import scoreP2
from scoreP3 import scoreP3


def extractZip(path, dest, flatten=True):
    """
    Extract a zip file, optionally flattening it into a single directory.
    """
    try:
        os.makedirs(dest)
    except OSError:
        if not os.path.exists(dest):
            raise

    try:
        with zipfile.ZipFile(path) as zf:
            if flatten:
                for name in zf.namelist():
                    # Ignore Mac OS X metadata
                    if name.startswith('__MACOSX'):
                        continue
                    outName = os.path.basename(name)
                    # Skip directories
                    if not outName:
                        continue
                    out = os.path.join(dest, outName)
                    with open(out, 'wb') as ofh:
                        with zf.open(name) as ifh:
                            while True:
                                buf = ifh.read(65536)
                                if buf:
                                    ofh.write(buf)
                                else:
                                    break
            else:
                zf.extractall(dest)
    except zipfile.BadZipfile as e:
        raise ScoreException('Could not read ZIP file "%s": %s' %
                             (os.path.basename(path), str(e)))


def unzipAll(directory, delete=True):
    """
    Unzip all zip files in directory and optionally delete them.
    Return a list of the zip file names.
    """
    zipFiles = [f for f in os.listdir(directory)
                if f.lower().endswith('.zip')]
    for zipFile in zipFiles:
        zipPath = os.path.join(directory, zipFile)
        extractZip(zipPath, directory)
        if delete:
            os.remove(zipPath)
    return zipFiles


def scoreAll(args):
    # Unzip zip files contained in the input folders
    truthDir = args.groundtruth
    truthZipSubFiles = unzipAll(truthDir, delete=True)
    truthPath = None
    if truthZipSubFiles:
        truthPath = truthZipSubFiles[0]
    else:
        truthSubFiles = os.listdir(truthDir)
        if truthSubFiles:
            truthPath = truthSubFiles[0]

    if not truthPath:
        raise ScoreException(
            'Internal error: error reading ground truth folder: %s' % truthDir)

    testDir = args.submission
    unzipAll(testDir, delete=True)

    # Identify which phase this is, based on ground truth file name
    truthRe = re.match(
        r'^ISIC-2017_(?:Test_v2|Validation)_'
        r'Part([0-9])_GroundTruth\.(?:csv|zip)$',
        os.path.basename(truthPath))
    if not truthRe:
        raise ScoreException(
            'Internal error: could not parse ground truth file name: %s' %
            os.path.basename(truthPath))
    phaseNum = truthRe.group(1)
    if phaseNum == '1':
        scores = scoreP1(truthDir, testDir)
    elif phaseNum == '2':
        scores = scoreP2(truthDir, testDir)
    elif phaseNum == '3':
        scores = scoreP3(truthDir, testDir)
    else:
        raise ScoreException(
            'Internal error: unknown ground truth phase number: %s' %
            os.path.basename(truthPath))

    print(json.dumps(scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Submission scoring helper script')
    parser.add_argument('-g', '--groundtruth', required=True,
                        help='path to the ground truth folder')
    parser.add_argument('-s', '--submission', required=True,
                        help='path to the submission folder')
    args = parser.parse_args()

    try:
        scoreAll(args)
    except ScoreException as e:
        covalicErrorPrefix = 'covalic.error: '
        print(covalicErrorPrefix + str(e), file=sys.stderr)
        exit(1)
