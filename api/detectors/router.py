"""POST /v1/vision/detection — CodeProject.AI object-detection contract."""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile

from api.detectors.yolo_detector import yolo_detector
from api.image_input import read_image_upload
from api.response import Timing, build_response, error_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/vision", tags=["object-detection"])

MODULE_ID = "ObjectDetection-YOLO26"
MODULE_NAME = "Object Detection (YOLO26)"
COMMAND = "detect"


@router.post("/detection")
async def detection(
    image: Annotated[UploadFile, File(description="Image to analyse")],
    min_confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.4,
    model: Annotated[str | None, Form(description="YOLO26 size variant: yolo26n/s/m/l/x")] = None,
) -> dict[str, Any]:
    with Timing() as timing:
        try:
            img = await read_image_upload(image, field_name="image")
        except Exception as exc:  # noqa: BLE001
            logger.warning("/v1/vision/detection rejected: %s", exc)
            return error_response(str(exc), module_id=MODULE_ID, command=COMMAND, timing=timing)

        logger.info(
            "/v1/vision/detection request: bytes=ok min_confidence=%.2f model=%s",
            min_confidence, model,
        )

        with timing.inference():
            predictions = yolo_detector.detect(img, min_confidence=min_confidence, variant=model)

    logger.info(
        "/v1/vision/detection done: count=%d inference_ms=%d total_ms=%d",
        len(predictions), timing.inference_ms, timing.process_ms,
    )
    return build_response(
        predictions,
        module_id=MODULE_ID,
        module_name=MODULE_NAME,
        command=COMMAND,
        timing=timing,
    )
