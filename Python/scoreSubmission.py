#!/usr/bin/env python

from __future__ import print_function

import argparse
import json
import os
import re
import sys
import zipfile

from PIL import Image
import numpy as np


def extractZip(path, dest, flatten=True):
    """
    Extract a zip file, optionally flattening it into a single directory.
    """
    try:
        os.makedirs(dest)
    except OSError:
        if not os.path.exists(dest):
            raise

    with zipfile.ZipFile(path) as zf:
        if flatten:
            for name in zf.namelist():
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


def matchInputFile(truthFile, testDir):
    # truthFile ~= 'ISIC_0000003_Segmentation.png'
    for testFile in os.listdir(testDir):
        truthFileId = truthFile.split('_')[1]
        if truthFileId in testFile:
            testPath = os.path.join(testDir, testFile)
            return testPath

    raise Exception('No matching submission image for: %s' % truthFile)


def loadImage(imagePath, rsize=-1):
    try:
        image = Image.open(imagePath)
    except:
        raise Exception('Could not decode image: %s' % imagePath)

    if image.mode != 'L':
        raise Exception('Image %s is not single-channel (grayscale).' %
                        os.path.basename(imagePath))

    return np.array(image)


def runScoringP1(truthPath, testPath):
    truthImage = loadImage(truthPath)
    testImage = loadImage(testPath)

    if testImage.shape[0:2] != truthImage.shape[0:2]:
        raise Exception('Image %s has dimensions %s; expected %s.' %
                        (os.path.basename(testPath), testImage.shape[0:2],
                         truthImage.shape[0:2]))

    truthBinaryImage = (truthImage > 128)
    testBinaryImage = (testImage > 128)  # TODO: make dynamic

    truthBinaryNegativeImage = 1 - truthBinaryImage
    testBinaryNegativeImage = 1 - testBinaryImage

    truthPixelSum = float(np.sum(truthBinaryImage))
    testPixelSum = float(np.sum(testBinaryImage))

    truePositive = float(np.sum(np.logical_and(truthBinaryImage,
                                               testBinaryImage)))
    trueNegative = float(np.sum(np.logical_and(truthBinaryNegativeImage,
                                               testBinaryNegativeImage)))
    falsePositive = float(np.sum(np.logical_and(truthBinaryNegativeImage,
                                                testBinaryImage)))
    falseNegative = float(np.sum(np.logical_and(truthBinaryImage,
                                                testBinaryNegativeImage)))

    metrics = [
        {
            'name': 'accuracy',
            'value': (truePositive + trueNegative) /
                     (truePositive + trueNegative + falsePositive + falseNegative)
        },
        {
            'name': 'jaccard',
            'value': (truePositive) / (truePositive + falseNegative + falsePositive)
        },
        {
            'name': 'dice',
            'value': (2 * truePositive) / (truthPixelSum + testPixelSum)
        },
        {
            'name': 'sensitivity',
            'value': (truePositive) / (truePositive + falseNegative)
        },
        {
            'name': 'specificity',
            'value': (trueNegative) / (trueNegative + falsePositive)
        }
    ]
    return metrics


def scoreAll(args):
    truthDir = '/covalic/Data/ISBI2016_ISIC_Part1_Test_GroundTruth'

    # Unzip the input files into appropriate folders
    testZipPath = args.submission
    testBaseDir = os.path.dirname(testZipPath)
    testDir = os.path.join(testBaseDir, 'submission')
    extractZip(testZipPath, testDir)

    # Unzip any zip files that were contained in the submission zip file
    zipFiles = [f for f in os.listdir(testDir) if f.lower().endswith('.zip')]
    for zipFile in zipFiles:
        zipFile = os.path.join(testDir, zipFile)
        extractZip(zipFile, testDir)

    # Iterate over each file and call scoring executable on the pair
    scores = []
    for truthFile in os.listdir(truthDir):
        # truthFile ~= 'ISIC_0000003_Segmentation.png'
        try:
            testPath = matchInputFile(truthFile, testDir)
            truthPath = os.path.join(truthDir, truthFile)

            datasetName = truthFile.rsplit('_', 1)[0]
            metrics = runScoringP1(truthPath, testPath)
        except Exception as e:
            print(str(e), file=sys.stderr)
            # TODO: Don't fail completely
            raise

        scores.append({
            'dataset': datasetName,
            'metrics': metrics
        })

    print(json.dumps(scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Submission scoring helper script')
    # parser.add_argument('-g', '--groundtruth', required=True,
    #                     help='path to the ground truth zip file')
    parser.add_argument('-s', '--submission', required=True,
                        help='path to the submission zip file')
    args = parser.parse_args()

    scoreAll(args)
