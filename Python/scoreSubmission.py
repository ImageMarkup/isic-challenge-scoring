#!/usr/bin/env python
# coding=utf-8

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


def unzipAll(directory, delete=True):
    """
    Unzip all zip files in directory and optionally delete them.
    Return a list of the zip file filenames.
    """
    zipFiles = [f for f in os.listdir(directory)
                if f.lower().endswith('.zip')]
    for zipFile in zipFiles:
        zipPath = os.path.join(directory, zipFile)
        extractZip(zipPath, directory)
        if delete:
            os.remove(zipPath)
    return zipFiles


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
        raise Exception('Could not decode image: %s' %
                        os.path.basename(imagePath))

    if image.mode != 'L':
        raise Exception('Image %s is not single-channel (grayscale).' %
                        os.path.basename(imagePath))

    image = np.array(image)

    imageValues = set(np.unique(image))
    if imageValues <= {0, 255}:
        # Expected values
        pass
    elif len(imageValues) <= 2:
        # Binary image with high value other than 255 can be corrected
        highValue = (imageValues - {0}).pop()
        image /= highValue
        image *= 255
        if set(np.unique(image)) > {0, 255}:
            raise Exception('Image %s contains values other than 0 and 255.' %
                            os.path.basename(imagePath))
    else:
        raise Exception('Image %s contains values other than 0 and 255.' %
                        os.path.basename(imagePath))

    return image


def scoreP1Image(truthPath, testPath):
    truthImage = loadImage(truthPath)
    testImage = loadImage(testPath)

    if testImage.shape[0:2] != truthImage.shape[0:2]:
        raise Exception('Image %s has dimensions %s; expected %s.' %
                        (os.path.basename(testPath), testImage.shape[0:2],
                         truthImage.shape[0:2]))

    truthBinaryImage = (truthImage > 128)
    testBinaryImage = (testImage > 128)

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


def scoreP1(truthDir, testDir):
    # Iterate over each file and call scoring executable on the pair
    scores = []
    for truthFile in sorted(os.listdir(truthDir)):
        # truthFile ~= 'ISIC_0000003_Segmentation.png'
        try:
            testPath = matchInputFile(truthFile, testDir)
            truthPath = os.path.join(truthDir, truthFile)

            datasetName = truthFile.rsplit('_', 1)[0]
            metrics = scoreP1Image(truthPath, testPath)
        except Exception as e:
            # print(str(e), file=sys.stderr)
            # TODO: Don't fail completely
            raise

        scores.append({
            'dataset': datasetName,
            'metrics': metrics
        })

    return scores


def scoreP2(truthDir, testDir):
    return []


def scoreP3(truthDir, testDir):
    return []


def scoreAll(args):
    # Unzip the input files into appropriate folders
    truthZipPath = args.groundtruth
    truthBaseDir = os.path.dirname(truthZipPath)
    truthDir = os.path.join(truthBaseDir, 'groundtruth')
    extractZip(truthZipPath, truthDir)
    truthZipSubFiles = unzipAll(truthDir, delete=True)
    if not truthZipSubFiles:
        raise Exception('Internal error: error reading ground truth file: %s'
                        % os.path.basename(truthZipPath))
    truthZipPath = truthZipSubFiles[0]

    testZipPath = args.submission
    testBaseDir = os.path.dirname(testZipPath)
    testDir = os.path.join(testBaseDir, 'submission')
    extractZip(testZipPath, testDir)
    unzipAll(testDir, delete=True)

    # Identify which phase this is, based on ground truth file name
    truthZipRe = re.match(r'^ISBI2016_ISIC_Part([0-9])_Test_GroundTruth\.zip$',
                          os.path.basename(truthZipPath))
    if not truthZipRe:
        raise Exception('Internal error: could not parse ground truth file '
                        'name: %s' % os.path.basename(truthZipPath))
    phaseNum = int(truthZipRe.group(1))
    if phaseNum == 1:
        scores = scoreP1(truthDir, testDir)
    elif phaseNum == 2:
        scores = scoreP2(truthDir, testDir)
    elif phaseNum == 3:
        scores = scoreP3(truthDir, testDir)
    else:
        raise Exception('Internal error: unknown ground truth phase number: %s' %
                        os.path.basename(truthZipPath))

    print(json.dumps(scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Submission scoring helper script')
    parser.add_argument('-g', '--groundtruth', required=True,
                        help='path to the ground truth zip file')
    parser.add_argument('-s', '--submission', required=True,
                        help='path to the submission zip file')
    args = parser.parse_args()

    try:
        scoreAll(args)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
