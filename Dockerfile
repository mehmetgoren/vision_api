# --- Base stage with Tesseract OCR ---
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    WEBSERVER_ENVIRONMENT=Container

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        tzdata \
        curl \
        git \
        build-essential \
        python3-dev \
        tesseract-ocr \
        tesseract-ocr-all \
        libtesseract-dev \
        libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .
RUN pip install .

EXPOSE 8014

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8014/health')"

CMD ["uvicorn", "api.methods:app", "--host", "0.0.0.0", "--port", "8014"]
