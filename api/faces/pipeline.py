"""InsightFace pipeline: SCRFD detector + ArcFace embeddings (buffalo_l)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)


_INSIGHTFACE_MISSING_DETAIL = (
    "InsightFace not installed. Run `uv sync` (or use the Docker image) "
    "to pull `insightface` and `onnxruntime-gpu`."
)


def _insightface_home() -> Path:
    home = Path(os.environ.get("INSIGHTFACE_HOME") or os.environ.get("MODELS_DIR", "models"))
    if "INSIGHTFACE_HOME" not in os.environ:
        home = home / "insightface"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["INSIGHTFACE_HOME"] = str(home)
    return home


class FaceEmbedding:
    """Minimal struct returned by FacePipeline.analyse()."""

    __slots__ = ("bbox", "det_score", "embedding")

    def __init__(self, bbox: tuple[int, int, int, int], det_score: float, embedding: np.ndarray) -> None:
        self.bbox = bbox
        self.det_score = det_score
        self.embedding = embedding


class FacePipeline:
    """Lazy wrapper around insightface.app.FaceAnalysis(buffalo_l).

    buffalo_l ships an SCRFD-10G face detector and an ArcFace-R100 embedding
    network. Embeddings are L2-normalised 512-d float32 vectors, so cosine
    similarity reduces to a dot product downstream.
    """

    def __init__(self, det_size: tuple[int, int] = (640, 640)) -> None:
        self.det_size = det_size
        self._app: Any = None

    def _get_app(self) -> Any:
        if self._app is not None:
            return self._app

        try:
            from insightface.app import FaceAnalysis  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning("insightface not installed: %s", exc)
            raise HTTPException(status_code=503, detail=_INSIGHTFACE_MISSING_DETAIL) from exc

        _insightface_home()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0, det_size=self.det_size)
        except Exception as exc:
            logger.warning("CUDA InsightFace init failed (%s); retrying CPU-only", exc)
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=self.det_size)

        self._app = app
        logger.info("InsightFace ready (det_size=%s, providers=%s)", self.det_size, providers)
        return app

    def warmup(self) -> None:
        _ = self._get_app()

    def analyse(self, image: np.ndarray, min_confidence: float = 0.0) -> list[FaceEmbedding]:
        app = self._get_app()
        faces = app.get(image)
        out: list[FaceEmbedding] = []
        for f in faces:
            det_score = float(getattr(f, "det_score", 0.0))
            if det_score < min_confidence:
                continue
            bbox = f.bbox.astype(int).tolist()
            embedding = np.asarray(f.normed_embedding, dtype=np.float32)
            out.append(
                FaceEmbedding(
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    det_score=det_score,
                    embedding=embedding,
                )
            )
        return out


face_pipeline = FacePipeline()
