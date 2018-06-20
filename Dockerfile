# Build container to create virtual environment with required packages
FROM ubuntu:18.04 AS venv_builder
RUN apt-get update && \
    apt-get install -y \
      python3 \
      python3-venv \
      python3-pip
WORKDIR /covalic
RUN python3 -m venv ./venv && \
    ./venv/bin/pip --no-cache-dir install \
      numpy \
      pillow \
      scipy \
      scikit-learn

# Mininal-size run container
FROM ubuntu:18.04
RUN apt-get update && \
    apt-get install -y \
      python3
WORKDIR /covalic
COPY --from=venv_builder /covalic .
COPY isic_challenge_scoring ./isic_challenge_scoring
ENTRYPOINT ["./venv/bin/python", "-m", "isic_challenge_scoring"]
