FROM python:3.7

RUN pip install tox \
  # Install Git LFS
  && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
  && apt-get install git-lfs
