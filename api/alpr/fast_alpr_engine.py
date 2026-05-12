"""License-plate recognition via fast-alpr (YOLOv9 detector + fast-plate-ocr).

Replaces CodeProject.AI's plate_reader module. Same on-the-wire contract,
materially better accuracy, fully GPU-accelerated through onnxruntime.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)


_FAST_ALPR_MISSING_DETAIL = (
    "fast-alpr not installed. Run `uv sync` to pull `fast-alpr[onnx-gpu]`."
)


class FastAlprEngine:
    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        ocr_model: str = "cct-xs-v2-global-model",
    ) -> None:
        self.detector_model = detector_model
        self.ocr_model = ocr_model
        self._alpr: Any = None

    def _get_alpr(self) -> Any:
        if self._alpr is not None:
            return self._alpr

        try:
            from fast_alpr import ALPR  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning("fast-alpr not installed: %s", exc)
            raise HTTPException(status_code=503, detail=_FAST_ALPR_MISSING_DETAIL) from exc

        self._alpr = ALPR(
            detector_model=self.detector_model,
            ocr_model=self.ocr_model,
        )
        logger.info(
            "FastALPR ready (detector=%s ocr=%s)",
            self.detector_model, self.ocr_model,
        )
        return self._alpr

    def warmup(self) -> None:
        _ = self._get_alpr()

    @staticmethod
    def _coerce_bbox(detection: Any) -> tuple[int, int, int, int] | None:
        """fast-alpr's result objects expose detection.bounding_box with x1/y1/x2/y2.
        Defensive: support both attribute and dict shapes, plus a tuple/list fallback.
        """
        bbox = getattr(detection, "bounding_box", None)
        if bbox is None and isinstance(detection, dict):
            bbox = detection.get("bounding_box")
        if bbox is None:
            return None
        x1 = getattr(bbox, "x1", None) or getattr(bbox, "x_min", None)
        y1 = getattr(bbox, "y1", None) or getattr(bbox, "y_min", None)
        x2 = getattr(bbox, "x2", None) or getattr(bbox, "x_max", None)
        y2 = getattr(bbox, "y2", None) or getattr(bbox, "y_max", None)
        if x1 is None and isinstance(bbox, dict):
            x1 = bbox.get("x1") or bbox.get("x_min")
            y1 = bbox.get("y1") or bbox.get("y_min")
            x2 = bbox.get("x2") or bbox.get("x_max")
            y2 = bbox.get("y2") or bbox.get("y_max")
        if x1 is None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None
        return int(x1), int(y1), int(x2), int(y2)

    def recognize(
        self,
        image: np.ndarray,
        min_confidence: float = 0.4,
    ) -> list[dict[str, Any]]:
        alpr = self._get_alpr()
        results = alpr.predict(image)

        predictions: list[dict[str, Any]] = []
        for r in results or []:
            bbox = self._coerce_bbox(r)
            if bbox is None:
                continue

            det_conf = float(
                getattr(r, "detection_confidence", None)
                or getattr(getattr(r, "detection", None), "confidence", 0.0)
                or 0.0
            )
            ocr = getattr(r, "ocr", None)
            text = (getattr(ocr, "text", None) or "").strip() if ocr is not None else ""
            ocr_conf = float(getattr(ocr, "confidence", 0.0) or 0.0) if ocr is not None else 0.0

            # Final confidence = OCR confidence if available, else detector's.
            confidence = ocr_conf if ocr_conf > 0.0 else det_conf
            if confidence < min_confidence:
                continue

            x1, y1, x2, y2 = bbox
            predictions.append(
                {
                    "confidence": round(confidence, 4),
                    "label": text or "plate",
                    "plate": text,
                    "detection_confidence": round(det_conf, 4),
                    "ocr_confidence": round(ocr_conf, 4),
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                }
            )

        return predictions


fast_alpr_engine = FastAlprEngine()
