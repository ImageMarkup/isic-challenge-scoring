FROM ubuntu:14.04
MAINTAINER Brian Helba <brian.helba@kitware.com>

# Install system prerequisites
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    freeglut3-dev \
    git \
    mesa-common-dev \
    python \
    python-pip \
    python-pil \
    libpython-dev \
    liblapack-dev \
    gfortran

RUN pip install numpy scipy scikit-learn

RUN mkdir /covalic

COPY Python /covalic/Python

ENTRYPOINT ["python", "/covalic/Python/scoreSubmission.py"]
