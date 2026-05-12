"""POST /v1/vision/scene — Places365 scene classifier."""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, UploadFile

from api.image_input import read_image_upload
from api.response import Timing, build_response, error_response
from api.scene.places365 import scene_classifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/vision", tags=["scene"])

MODULE_ID = "Scene-Places365"
MODULE_NAME = "Scene Classification (Places365)"
COMMAND = "classify"


@router.post("/scene")
async def scene(
    image: Annotated[UploadFile, File(description="Image to classify")],
) -> dict[str, Any]:
    with Timing() as timing:
        try:
            img = await read_image_upload(image, field_name="image")
        except Exception as exc:  # noqa: BLE001
            return error_response(str(exc), module_id=MODULE_ID, command=COMMAND, timing=timing)

        with timing.inference():
            result = scene_classifier.classify(img)

    logger.info(
        "/v1/vision/scene done: label=%s confidence=%.3f total_ms=%d",
        result["label"], result["confidence"], timing.process_ms,
    )
    return build_response(
        None,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=COMMAND,
        timing=timing,
        extra={
            "label": result["label"],
            "confidence": result["confidence"],
            "top_k": result["top_k"],
        },
    )
