FROM ubuntu:14.04
MAINTAINER Brian Helba <brian.helba@kitware.com>

RUN mkdir /covalic

COPY Python /covalic/Python
COPY Data /covalic/Data

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

RUN pip install numpy scipy

ENTRYPOINT ["python", "/covalic/Python/scoreSubmission.py"]
