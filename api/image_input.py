"""Shared UploadFile -> numpy BGR helper.

InsightFace, fast-alpr, OpenCV, and Ultralytics all consume BGR np.uint8 arrays
(OpenCV's native layout). Pillow gives us RGB by default — we flip channels
here so callers can stay backend-agnostic.
"""
from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


async def read_image_upload(file: UploadFile, *, field_name: str = "image") -> np.ndarray:
    """Read a multipart UploadFile and decode it as a BGR np.uint8 array.

    Raises HTTPException(400) for an empty upload or unreadable image.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"'{field_name}' is empty")
    return decode_image_bytes(raw, field_name=field_name)


def decode_image_bytes(raw: bytes, *, field_name: str = "image") -> np.ndarray:
    """Decode raw bytes to a BGR np.uint8 array.

    EXIF orientation is honored. RGBA/grayscale inputs are converted to RGB
    before the BGR flip so downstream code can assume HxWx3 BGR.
    """
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("decode_image_bytes(%s) failed: %s", field_name, exc)
        raise HTTPException(
            status_code=400,
            detail=f"'{field_name}' is not a decodable image: {exc}",
        ) from exc

    rgb = np.array(image.convert("RGB"))
    bgr: np.ndarray = rgb[:, :, ::-1].copy()
    return bgr
