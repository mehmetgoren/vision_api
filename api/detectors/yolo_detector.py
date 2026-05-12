"""YOLO26 (Ultralytics) object detector.

Default checkpoint is yolo26s.pt — good speed/accuracy balance for general
COCO classes. Callers can pick a different size variant per-request via the
`model` form field (n/s/m/l/x). Models are loaded lazily and cached per
variant so the first call to a given size pays the download cost, subsequent
calls are warm.

Weights are downloaded to MODELS_DIR/ultralytics (default ./models/ultralytics)
on first use. Pre-download in the Dockerfile or mount a volume to avoid the
cold-start hit.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from api.detectors.base import BaseDetector, Prediction

logger = logging.getLogger(__name__)


_YOLO_MISSING_DETAIL = (
    "Ultralytics not installed. Run `uv sync` (or use the Docker image) "
    "to pull `ultralytics` and `torch` from the cu126 index."
)

# Public YOLO26 size variants — callers can request any of these.
ALLOWED_VARIANTS: tuple[str, ...] = (
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
)


def _models_dir() -> Path:
    base = Path(os.environ.get("MODELS_DIR", "models")) / "ultralytics"
    base.mkdir(parents=True, exist_ok=True)
    return base


class YoloDetector(BaseDetector):
    def __init__(self, default_variant: str = "yolo26s", device: str | None = None) -> None:
        if default_variant not in ALLOWED_VARIANTS:
            raise ValueError(
                f"Unknown YOLO variant {default_variant!r}; expected one of {ALLOWED_VARIANTS}"
            )
        self.default_variant = default_variant
        self.device = device
        self._models: dict[str, Any] = {}

    def _resolve_variant(self, requested: str | None) -> str:
        if not requested:
            return self.default_variant
        cleaned = requested.strip().lower()
        if cleaned.endswith(".pt"):
            cleaned = cleaned[:-3]
        if cleaned not in ALLOWED_VARIANTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model {requested!r}; expected one of {ALLOWED_VARIANTS}",
            )
        return cleaned

    def _get_model(self, variant: str) -> Any:
        cached = self._models.get(variant)
        if cached is not None:
            return cached

        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning("ultralytics not installed: %s", exc)
            raise HTTPException(status_code=503, detail=_YOLO_MISSING_DETAIL) from exc

        weights_path = _models_dir() / f"{variant}.pt"
        if not weights_path.exists():
            # Ultralytics downloads to cwd when handed a bare filename. Briefly
            # chdir into the cache so the .pt lands in MODELS_DIR/ultralytics.
            old_cwd = Path.cwd()
            try:
                os.chdir(weights_path.parent)
                model = YOLO(f"{variant}.pt")
            finally:
                os.chdir(old_cwd)
        else:
            model = YOLO(str(weights_path))

        if self.device:
            try:
                model.to(self.device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("YOLO.to(%s) failed: %s", self.device, exc)

        self._models[variant] = model
        logger.info("YOLO model loaded: %s (weights=%s)", variant, weights_path)
        return model

    def warmup(self) -> None:
        _ = self._get_model(self.default_variant)

    def detect(
        self,
        image: np.ndarray,
        min_confidence: float = 0.4,
        variant: str | None = None,
    ) -> list[Prediction]:
        resolved = self._resolve_variant(variant)
        model = self._get_model(resolved)

        results = model.predict(
            image,
            conf=float(min_confidence),
            verbose=False,
        )

        predictions: list[Prediction] = []
        names: dict[int, str] = {}

        for r in results:
            names = getattr(r, "names", None) or names
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                conf_f = float(conf)
                if conf_f < min_confidence:
                    continue
                label = names.get(int(cls), str(int(cls))) if names else str(int(cls))
                predictions.append(
                    {
                        "confidence": round(conf_f, 4),
                        "label": str(label),
                        "x_min": int(x1),
                        "y_min": int(y1),
                        "x_max": int(x2),
                        "y_max": int(y2),
                    }
                )

        return predictions


yolo_detector = YoloDetector()
