FROM python:3.13

WORKDIR /usr/src/isic-challenge-scoring

RUN --mount=source=.,target=. \
    pip install --no-cache-dir .

ENTRYPOINT ["isic-challenge-scoring"]
