# syntax=docker/dockerfile:1.7
#
# vision_api OCR runtime.
#
# Base: nvidia/cuda 12.6 + cuDNN — paddlepaddle-gpu==3.2.1 (cu126 wheels)
# needs CUDA libs at runtime. Run with `docker run --gpus all ...` to use
# the GPU; without --gpus, Paddle requests will fail but the Kreuzberg
# (tesseract) engine still works on CPU.
#
# Build:   docker build -t vision_api .
# Run:     docker run --gpus all -p 8014:8014 vision_api

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    WEBSERVER_ENVIRONMENT=Container \
    FACE_REGISTRY_DIR=/app/data \
    MODELS_DIR=/app/models \
    INSIGHTFACE_HOME=/app/models/insightface \
    TORCH_HOME=/app/models/torch \
    PLACES365_WEIGHTS_DIR=/app/models/places365

# System deps:
#  - tesseract + lang packs — Kreuzberg OCR backend
#  - libgomp1, libglib2.0-0, libgl1 — paddle / opencv runtime
#  - libjpeg/png/tiff/webp — image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget tzdata \
        tesseract-ocr tesseract-ocr-all \
        libtesseract-dev libleptonica-dev \
        libglib2.0-0 libgomp1 libgl1 \
        libjpeg-turbo8 libpng16-16 libtiff5 libwebp7 \
        libfontconfig1 libfreetype6 \
        libssl3 zlib1g \
    && rm -rf /var/lib/apt/lists/*

# uv (Python project + interpreter manager).
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install Python deps first (cacheable layer). uv pulls Python itself
# since 3.12+ isn't in the Ubuntu 22.04 repos. paddlepaddle-gpu is
# sourced from the paddle cu126 index (configured in pyproject.toml).
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

# App source.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# Persistent volumes: /app/data (face registry SQLite + embeddings) and
# /app/models (YOLO26 / InsightFace / Places365 / fast-alpr weights).
RUN mkdir -p /app/data /app/models

EXPOSE 8014

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8014/health')"

CMD ["uvicorn", "api.methods:app", "--host", "0.0.0.0", "--port", "8014"]
