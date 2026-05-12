from __future__ import annotations

import binascii
import logging
from enum import Enum
from typing import Annotated, Any

import pybase64
from fastapi import FastAPI, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.ocrs import BaseOcr, kreuzberg_ocr, paddle_ocr

logger = logging.getLogger(__name__)

app = FastAPI(title="OCR API", version="0.1.0")


class OcrEngine(str, Enum):
    KREUZBERG = "Kreuzberg"
    PADDLE = "Paddle"


_ENGINES: dict[OcrEngine, BaseOcr] = {
    OcrEngine.KREUZBERG: kreuzberg_ocr,
    OcrEngine.PADDLE: paddle_ocr,
}


class OcrRequest(BaseModel):
    image_bytes: str | None = Field(
        None,
        description="Image bytes, base64-encoded (optionally with a data URI prefix). Checked first.",
    )
    image_file: UploadFile | None = Field(
        None,
        description=(
            "File upload. Used when image_bytes is empty. Accepts any "
            "format the selected engine supports — images (PNG/JPEG/TIFF/"
            "BMP/WebP/GIF), PDFs, and (Kreuzberg only) office docs, HTML, "
            "email, archives, etc. Non-image inputs return per-line text "
            "predictions with zero bbox."
        ),
    )
    min_confidence: float = Field(0.4, ge=0.0, le=1.0)
    engine: OcrEngine = Field(
        OcrEngine.KREUZBERG,
        description="OCR engine to use. Default: Kreuzberg.",
    )
    improve_accuracy: bool = Field(
        False,
        description=(
            "Paddle-only HQ mode. When true, routes Paddle to the PaddleOCRVL "
            "pipeline (doc orientation/unwarp/seal recognition + OCR-for-image-block). "
            "Slower. Ignored by Kreuzberg."
        ),
    )


def _decode_base64_image(data: str) -> bytes:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return pybase64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc


@app.post("/ocr")
async def ocr_endpoint(payload: Annotated[OcrRequest, Form()]) -> dict[str, Any]:
    raw: bytes | None = None
    filename: str | None = None
    if payload.image_bytes:
        raw = _decode_base64_image(payload.image_bytes)
    elif payload.image_file is not None:
        raw = await payload.image_file.read()
        filename = payload.image_file.filename

    if not raw:
        logger.warning("/ocr called with neither image_bytes nor image_file")
        raise HTTPException(
            status_code=400,
            detail="Either image_bytes (base64 string) or image_file (file upload) must be provided.",
        )

    return _ENGINES[payload.engine].ocr(
        raw,
        payload.min_confidence,
        improve_accuracy=payload.improve_accuracy,
        filename=filename,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
