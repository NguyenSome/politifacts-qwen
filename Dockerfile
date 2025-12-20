FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         bash python3.11 python3.11-venv python3.11-distutils python3.11-dev \
         curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir nvidia-ml-py3 huggingface_hub hf_transfer

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade --ignore-installed -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

COPY data/ ./data/
COPY src/ ./src/
COPY configs ./configs/
RUN mkdir -p ./logs

RUN chmod +x ./src/entrypoint.sh

ENTRYPOINT ["/app/src/entrypoint.sh"]
