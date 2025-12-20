FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash libglib2.0-0 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir nvidia-ml-py3 huggingface_hub hf_transfer

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
         --extra-index-url https://download.pytorch.org/whl/cu124

COPY data/ ./data/
COPY src/ ./src/
COPY configs ./configs/
COPY logs/ ./logs/

RUN chmod +x ./src/entrypoint.sh

ENTRYPOINT ["/app/src/entrypoint.sh"]