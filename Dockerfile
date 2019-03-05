FROM python:3.7

WORKDIR /usr/src/isic-challenge-scoring

COPY ./setup.* ./
COPY ./isic_challenge_scoring ./isic_challenge_scoring

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "isic_challenge_scoring"]
