from __future__ import annotations

import io
import logging
import mimetypes
from typing import Any

import pytesseract
from PIL import Image, UnidentifiedImageError
from kreuzberg import (
    ExtractionConfig,
    OcrConfig,
    TesseractConfig,
    extract_bytes_sync,
)

from api.ocrs.base import BaseOcr

logger = logging.getLogger(__name__)


_MAGIC_BYTES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF",        "application/pdf"),
    (b"\x89PNG\r\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a",      "image/gif"),
    (b"GIF89a",      "image/gif"),
    (b"BM",          "image/bmp"),
    (b"II*\x00",     "image/tiff"),
    (b"MM\x00*",     "image/tiff"),
    (b"RIFF",        "image/webp"),  # weak — narrowed below
)


def _detect_mime(image_bytes: bytes, filename: str | None) -> str:
    if filename:
        guess, _ = mimetypes.guess_type(filename)
        if guess:
            return guess
    head = image_bytes[:16]
    for prefix, mime in _MAGIC_BYTES:
        if head.startswith(prefix):
            if prefix == b"RIFF" and b"WEBP" not in image_bytes[:16]:
                continue
            return mime
    return "application/octet-stream"


class KreuzbergOcr(BaseOcr):
    def ocr(
        self,
        image_bytes: bytes,
        min_confidence: float = 0.4,
        improve_accuracy: bool = False,
        filename: str | None = None,
    ) -> dict[str, Any]:
        del improve_accuracy  # Kreuzberg has no HQ mode toggle

        mime = _detect_mime(image_bytes, filename)
        logger.debug("Kreuzberg input mime=%s filename=%r", mime, filename)

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

        result = extract_bytes_sync(image_bytes, mime, config)
        logger.info(
            "Kreuzberg extracted %d characters (mime=%s)",
            len(result.content), mime,
        )

        predictions: list[dict[str, Any]] = []

        if mime.startswith("image/"):
            try:
                image = Image.open(io.BytesIO(image_bytes))
                tsv_data = pytesseract.image_to_data(
                    image, output_type=pytesseract.Output.DICT
                )
            except (UnidentifiedImageError, OSError) as exc:
                logger.warning("Pillow could not open image for bbox extraction: %s", exc)
                tsv_data = None

            if tsv_data is not None:
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

        # Non-image input (PDF, docx, ...): no per-region bbox available;
        # emit one prediction per non-empty text line with a zero bbox.
        predictions.append(
            {
                "confidence": 1.0,
                "label": result.content,
                "x_min": 0,
                "y_min": 0,
                "x_max": 0,
                "y_max": 0,
            }
        )
        return {"predictions": predictions}


kreuzberg_ocr = KreuzbergOcr()
