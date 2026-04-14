FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DOWNLOAD_WEIGHTS_AT_BUILD=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PYTHONPATH=/app:/app/musetalk/utils \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860 \
    USE_FLOAT16=1 \
    DOWNLOAD_WEIGHTS=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install \
    --retries 10 \
    --timeout 300 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --retries 10 --timeout 300 -r /app/requirements.txt
RUN python3 -m pip install --retries 10 --timeout 300 --no-cache-dir -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.0.1"
RUN mim install "mmdet==3.1.0"
RUN python3 -m pip install --retries 10 --timeout 300 "mmpose==1.3.2"

COPY . /app

RUN chmod +x /app/docker-entrypoint.sh /app/serverless-entrypoint.sh /app/download_weights.sh

RUN if [ "$DOWNLOAD_WEIGHTS_AT_BUILD" = "1" ]; then bash /app/download_weights.sh; fi

EXPOSE 7860

ENTRYPOINT ["/app/serverless-entrypoint.sh"]
