# ISIC: Skin Lesion Analysis Towards Melanoma Detection Scoring

[![CircleCI](https://circleci.com/gh/ImageMarkup/isic-challenge-scoring.svg?style=svg)](https://circleci.com/gh/ImageMarkup/isic-challenge-scoring)
[![GitHub license](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://raw.githubusercontent.com/ImageMarkup/isic-challenge-scoring/master/LICENSE)

Automated scoring code for the [ISIC Challenge](http://challenge.isic-archive.com).

## Installation
### Python
Python version >= 3.7 is required.
```bash
pip install isic-challenge-scoring
```

### Docker
```bash
docker pull isic/isic-challenge-scoring:latest
```

## Usage
### Python
#### Segmentation (Task 1)
```bash
isic-challenge-scoring segmentation /path/to/ISIC_GroundTruth/ /path/to/ISIC_predictions/
```

#### Classification (Task 3)
```bash
isic-challenge-scoring classification /path/to/ISIC_GroundTruth.csv /path/to/ISIC_prediction.csv
```

### Docker
Since the application requires read access to files, [Docker must mount](https://docs.docker.com/storage/bind-mounts/#use-a-read-only-bind-mount) them within the container; these examples use `--mount` to [prevent nonexistent host paths from being accidentally created](https://github.com/moby/moby/issues/13121).

#### Segmentation (Task 1)
```bash
docker run \
  --rm \
  --mount type=bind,source="/path/to/ISIC_GroundTruth/",destination=/root/gt/,readonly \
  --mount type=bind,source="/path/to/ISIC_predictions/",destination=/root/pred/,readonly \
  isic/isic-challenge-scoring:latest \
  segmentation \
  /root/gt/ \
  /root/pred/
```

#### Classification (Task 3)
```bash
docker run \
  --rm \
  --mount type=bind,source="/path/to/ISIC_GroundTruth.csv",destination=/root/gt.csv,readonly \
  --mount type=bind,source="/path/to/ISIC_prediction.csv",destination=/root/pred.csv,readonly \
  isic/isic-challenge-scoring:latest \
  classification \
  /root/gt.csv \
  /root/pred.csv
```
