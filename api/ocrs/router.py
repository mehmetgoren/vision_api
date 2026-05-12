"""POST /v1/vision/ocr and /ocr (legacy alias) — text recognition (Kreuzberg/Paddle)."""
from __future__ import annotations

import binascii
import logging
from enum import Enum
from typing import Annotated, Any

import pybase64
from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.ocrs import BaseOcr, kreuzberg_ocr, paddle_ocr
from api.pdf_detect import scan_pdf
from api.response import Timing, build_response, error_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ocr"])

MODULE_ID = "OCR-Vision"
MODULE_NAME = "Text Recognition (OCR)"
COMMAND = "ocr"


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


async def _ocr_impl(payload: OcrRequest) -> dict[str, Any]:
    with Timing() as timing:
        raw: bytes | None = None
        filename: str | None = None
        source: str = "none"
        try:
            if payload.image_bytes:
                raw = _decode_base64_image(payload.image_bytes)
                source = "image_bytes"
            elif payload.image_file is not None:
                raw = await payload.image_file.read()
                filename = payload.image_file.filename
                source = "image_file"
        except HTTPException as exc:
            return error_response(str(exc.detail), module_id=MODULE_ID, command=COMMAND, timing=timing)

        if not raw:
            logger.warning("/ocr rejected: neither image_bytes nor image_file provided")
            return error_response(
                "Either image_bytes (base64 string) or image_file (file upload) must be provided.",
                module_id=MODULE_ID,
                command=COMMAND,
                timing=timing,
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

        with timing.inference():
            result = _ENGINES[resolved].ocr(
                raw,
                payload.min_confidence,
                improve_accuracy=payload.improve_accuracy,
                filename=filename,
            )
        predictions = result.get("predictions", [])

    logger.info(
        "/ocr done: engine=%s predictions=%d inference_ms=%d total_ms=%d filename=%r",
        resolved.value, len(predictions), timing.inference_ms, timing.process_ms, filename,
    )
    return build_response(
        predictions,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=COMMAND,
        timing=timing,
        extra={"engine": resolved.value},
    )


@router.post("/v1/vision/ocr")
async def ocr(payload: Annotated[OcrRequest, Form()]) -> dict[str, Any]:
    return await _ocr_impl(payload)


@router.post("/ocr")
async def ocr_legacy(payload: Annotated[OcrRequest, Form()]) -> dict[str, Any]:
    """Legacy path kept for clients still using /ocr."""
    return await _ocr_impl(payload)
