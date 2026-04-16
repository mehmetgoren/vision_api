from __future__ import annotations

import binascii
import io
import logging
from typing import Any

import pybase64
import pytesseract
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from kreuzberg import (
    ExtractionConfig,
    OcrConfig,
    TesseractConfig,
    extract_bytes_sync,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="OCR API", version="0.1.0")


def ocr(image_bytes: bytes, min_confidence: float = 0.4) -> dict[str, Any]:
    """Run OCR on image bytes and return predictions with bounding boxes.

    Uses kreuzberg (tesseract backend) for OCR processing.
    Returns text regions with confidence scores and bounding box coordinates.
    """

    config = ExtractionConfig(
        ocr=OcrConfig(
            backend="tesseract",
            language="eng",
            tesseract_config=TesseractConfig(
                min_confidence=min_confidence,
                psm=3,
                oem=3,
            ),
        ),
        force_ocr=True,
    )

    # Use kreuzberg for the main OCR extraction
    result = extract_bytes_sync(image_bytes, "image/png", config)
    logger.info("Kreuzberg extracted %d characters", len(result.content))

    # Use pytesseract to get per-word bounding boxes with confidence
    image = Image.open(io.BytesIO(image_bytes))
    tsv_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    predictions = []
    n_items = len(tsv_data["text"])
    for i in range(n_items):
        text = tsv_data["text"][i].strip()
        conf = tsv_data["conf"][i]

        if conf == -1 or not text:
            continue

        confidence = conf / 100.0
        if confidence < min_confidence:
            continue

        x = tsv_data["left"][i]
        y = tsv_data["top"][i]
        w = tsv_data["width"][i]
        h = tsv_data["height"][i]

        predictions.append(
            {
                "confidence": round(confidence, 4),
                "label": text,
                "x_min": x,
                "y_min": y,
                "x_max": x + w,
                "y_max": y + h,
            }
        )

    return {"predictions": predictions}


class OcrRequest(BaseModel):
    base64_image: str = Field(..., description="Base64-encoded image bytes (optionally with data URI prefix)")
    min_confidence: float = Field(0.4, ge=0.0, le=1.0)


def _decode_base64_image(data: str) -> bytes:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return pybase64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

@app.post("/ocr")
async def ocr_endpoint(payload: OcrRequest) -> dict[str, Any]:
    image_bytes = _decode_base64_image(payload.base64_image)
    return ocr(image_bytes, payload.min_confidence)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
