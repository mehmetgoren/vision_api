from __future__ import annotations

import binascii
import logging
import time
from enum import Enum
from typing import Annotated, Any

import pybase64
from fastapi import FastAPI, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.alpr.router import router as alpr_router
from api.detectors.router import router as detection_router
from api.faces.router import router as face_router
from api.ocrs import BaseOcr, kreuzberg_ocr, paddle_ocr
from api.pdf_detect import scan_pdf
from api.scene.router import router as scene_router

logger = logging.getLogger(__name__)

app = FastAPI(title="Vision API", version="0.2.0")

app.include_router(detection_router)
app.include_router(face_router)
app.include_router(alpr_router)
app.include_router(scene_router)


class OcrEngine(str, Enum):
    AUTO = "Auto"
    KREUZBERG = "Kreuzberg"
    PADDLE = "Paddle"


_ENGINES: dict[OcrEngine, BaseOcr] = {
    OcrEngine.KREUZBERG: kreuzberg_ocr,
    OcrEngine.PADDLE: paddle_ocr,
}


def _resolve_engine(engine: OcrEngine, raw: bytes) -> OcrEngine:
    """Resolve OcrEngine.AUTO to a concrete engine based on the input bytes.

    PDFs that look scanned go to Paddle; everything else (native PDFs,
    images, office docs, etc.) goes to Kreuzberg. Non-AUTO selections pass
    through unchanged.
    """
    if engine != OcrEngine.AUTO:
        return engine
    return OcrEngine.PADDLE if scan_pdf(raw) else OcrEngine.KREUZBERG


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
        description=(
            "OCR engine to use. Default: Kreuzberg. 'Auto' picks Kreuzberg "
            "for images, office docs, and native PDFs, and Paddle for "
            "scanned PDFs."
        ),
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
    source: str
    if payload.image_bytes:
        raw = _decode_base64_image(payload.image_bytes)
        source = "image_bytes"
    elif payload.image_file is not None:
        raw = await payload.image_file.read()
        filename = payload.image_file.filename
        source = "image_file"
    else:
        source = "none"

    if not raw:
        logger.warning("/ocr rejected: neither image_bytes nor image_file provided")
        raise HTTPException(
            status_code=400,
            detail="Either image_bytes (base64 string) or image_file (file upload) must be provided.",
        )

    logger.info(
        "/ocr request: engine=%s source=%s filename=%r bytes=%d "
        "min_confidence=%.2f improve_accuracy=%s",
        payload.engine.value, source, filename, len(raw),
        payload.min_confidence, payload.improve_accuracy,
    )

    resolved = _resolve_engine(payload.engine, raw)
    if payload.engine == OcrEngine.AUTO:
        logger.info(
            "/ocr engine=Auto resolved to %s (filename=%r)",
            resolved.value, filename,
        )

    start = time.perf_counter()
    result = _ENGINES[resolved].ocr(
        raw,
        payload.min_confidence,
        improve_accuracy=payload.improve_accuracy,
        filename=filename,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "/ocr done: engine=%s predictions=%d elapsed_ms=%.1f filename=%r",
        resolved.value, len(result.get("predictions", [])), elapsed_ms, filename,
    )
    return result


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
