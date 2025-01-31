# ISIC: Skin Lesion Analysis Towards Melanoma Detection Scoring

[![PyPI](https://img.shields.io/pypi/v/isic-challenge-scoring)](https://pypi.org/project/isic-challenge-scoring/)
[![Docker Build Status](https://img.shields.io/docker/build/isic/isic-challenge-scoring)](https://hub.docker.com/r/isic/isic-challenge-scoring)

Automated scoring code for the [ISIC Challenge](http://challenge.isic-archive.com).

## Installation
### Python
Python version >= 3.13 is required.
```bash
pip install isic-challenge-scoring
```

### Docker
```bash
docker pull isic/isic-challenge-scoring:latest
```

## Usage
### Python
#### Segmentation (2016 Tasks 1 & 2B, 2017 Task 1, 2018 Tasks 1 & 2)
```bash
isic-challenge-scoring segmentation /path/to/ISIC_GroundTruth/ /path/to/ISIC_predictions/
```

#### Classification (2016 Tasks 3 & 3B, 2017 Task 3, 2018 Task 3, 2019 Tasks 1 & 2)
```bash
isic-challenge-scoring classification /path/to/ISIC_GroundTruth.csv /path/to/ISIC_prediction.csv
```

### Docker
Since the application requires read access to files, [Docker must mount](https://docs.docker.com/storage/bind-mounts/#use-a-read-only-bind-mount) them within the container; these examples use `--mount` to [prevent nonexistent host paths from being accidentally created](https://github.com/moby/moby/issues/13121).

#### Segmentation (2016 Tasks 1 & 2B, 2017 Task 1, 2018 Tasks 1 & 2)
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

#### Classification (2016 Tasks 3 & 3B, 2017 Task 3, 2018 Task 3, 2019 Tasks 1 & 2)
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
