from __future__ import annotations

import logging

from fastapi import FastAPI

from api.alpr.router import router as alpr_router
from api.detectors.router import router as detection_router
from api.faces.router import router as face_router
from api.ocrs.router import router as ocr_router
from api.scene.router import router as scene_router

logger = logging.getLogger(__name__)

app = FastAPI(title="Vision API", version="0.2.0")

app.include_router(detection_router)
app.include_router(face_router)
app.include_router(alpr_router)
app.include_router(scene_router)
app.include_router(ocr_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
