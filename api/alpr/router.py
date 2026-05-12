"""POST /v1/vision/alpr and /v1/image/alpr (legacy alias) — license plate reader."""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile

from api.alpr.fast_alpr_engine import fast_alpr_engine
from api.image_input import read_image_upload
from api.response import Timing, build_response, error_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["alpr"])

MODULE_ID = "ALPR-FastALPR"
MODULE_NAME = "License Plate Reader (fast-alpr)"
COMMAND = "alpr"


async def _alpr_impl(
    upload: UploadFile,
    min_confidence: float,
) -> dict[str, Any]:
    with Timing() as timing:
        try:
            img = await read_image_upload(upload, field_name="upload")
        except Exception as exc:  # noqa: BLE001
            return error_response(str(exc), module_id=MODULE_ID, command=COMMAND, timing=timing)

        with timing.inference():
            predictions = fast_alpr_engine.recognize(img, min_confidence=min_confidence)

    logger.info(
        "/alpr done: plates=%d inference_ms=%d total_ms=%d",
        len(predictions), timing.inference_ms, timing.process_ms,
    )
    return build_response(
        predictions,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=COMMAND,
        timing=timing,
    )


@router.post("/v1/vision/alpr")
async def alpr(
    upload: Annotated[UploadFile, File(description="Image to analyse")],
    min_confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.4,
) -> dict[str, Any]:
    return await _alpr_impl(upload, min_confidence)


@router.post("/v1/image/alpr")
async def alpr_legacy(
    upload: Annotated[UploadFile, File(description="Image to analyse")],
    min_confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.4,
) -> dict[str, Any]:
    """Legacy CodeProject.AI path kept for clients still using /v1/image/alpr."""
    return await _alpr_impl(upload, min_confidence)
