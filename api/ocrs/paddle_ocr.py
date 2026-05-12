from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from api.ocrs.base import BaseOcr

logger = logging.getLogger(__name__)


_PADDLE_MISSING_DETAIL = (
    "Paddle engine unavailable: paddleocr is not installed in this environment. "
    "Either build/run the Docker image (which installs paddleocr + paddlepaddle-gpu "
    "from the CUDA 12.6 index), or use engine=Kreuzberg."
)


class PaddleOcr(BaseOcr):
    def __init__(self, lang: str = "en", device: str | None = None) -> None:
        self.lang = lang
        self.device = device
        self._pipeline: Any = None
        self._pipeline_hq: Any = None

    def _get_pipeline(self, improve_accuracy: bool) -> Any:
        if improve_accuracy:
            if self._pipeline_hq is None:
                try:
                    from paddleocr import PaddleOCRVL
                except ImportError as exc:
                    logger.warning("paddleocr not installed: %s", exc)
                    raise HTTPException(status_code=503, detail=_PADDLE_MISSING_DETAIL) from exc

                self._pipeline_hq = PaddleOCRVL(
                    device=self.device,
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=True,
                    use_seal_recognition=True,
                    use_chart_recognition=False,
                    use_ocr_for_image_block=True,
                )
            return self._pipeline_hq
        if self._pipeline is None:
            try:
                from paddleocr import PaddleOCR as _PaddleOCR
            except ImportError as exc:
                logger.warning("paddleocr not installed: %s", exc)
                raise HTTPException(status_code=503, detail=_PADDLE_MISSING_DETAIL) from exc

            kwargs: dict[str, Any] = {
                "lang": self.lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
            }
            if self.device:
                kwargs["device"] = self.device
            self._pipeline = _PaddleOCR(**kwargs)
        return self._pipeline

    def warmup(self) -> None:
        _ = self._get_pipeline(False)

    @staticmethod
    def _input_for_predict(image_bytes: bytes, filename: str | None) -> tuple[Any, str | None]:
        """Return (input_for_predict, temp_path_to_cleanup).

        Paddle's predict() accepts numpy arrays for images and file paths for
        anything else (PDFs). Prefer the in-memory path for images so we don't
        hit the filesystem unnecessarily.
        """
        try:
            import numpy as np
        except ImportError as exc:
            logger.warning("numpy not installed: %s", exc)
            raise HTTPException(status_code=503, detail=_PADDLE_MISSING_DETAIL) from exc

        suffix = Path(filename).suffix if filename else ""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            return np.array(image.convert("RGB")), None
        except (UnidentifiedImageError, OSError):
            pass

        # Non-image (PDF, ...) — Paddle accepts a file path.
        tmp = NamedTemporaryFile(suffix=suffix or ".pdf", delete=False)
        try:
            tmp.write(image_bytes)
            tmp.flush()
        finally:
            tmp.close()
        return tmp.name, tmp.name

    def ocr(
        self,
        image_bytes: bytes,
        min_confidence: float = 0.4,
        improve_accuracy: bool = False,
        filename: str | None = None,
    ) -> dict[str, Any]:
        pipeline = self._get_pipeline(improve_accuracy)

        predict_input, cleanup_path = self._input_for_predict(image_bytes, filename)
        try:
            page_results = list(pipeline.predict(predict_input))
        finally:
            if cleanup_path:
                try:
                    os.unlink(cleanup_path)
                except OSError:
                    pass

        logger.info(
            "Paddle produced %d page result(s) (improve_accuracy=%s filename=%r)",
            len(page_results), improve_accuracy, filename,
        )

        predictions: list[dict[str, Any]] = []
        for res in page_results:
            data = res.json if hasattr(res, "json") else {}
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            # PaddleOCRVL nests the per-line OCR under overall_ocr_res
            if isinstance(data, dict) and "overall_ocr_res" in data:
                data = data["overall_ocr_res"]

            polys = data.get("rec_polys") or data.get("dt_polys") or []
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []

            for poly, text, score in zip(polys, texts, scores):
                label = str(text).strip()
                if not label:
                    continue
                confidence = float(score)
                if confidence < min_confidence:
                    continue

                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]

                predictions.append(
                    {
                        "confidence": round(confidence, 4),
                        "label": label,
                        "x_min": int(min(xs)),
                        "y_min": int(min(ys)),
                        "x_max": int(max(xs)),
                        "y_max": int(max(ys)),
                    }
                )

        return {"predictions": predictions}


paddle_ocr = PaddleOcr()
