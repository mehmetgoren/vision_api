from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseOcr(ABC):
    @abstractmethod
    def ocr(
        self,
        image_bytes: bytes,
        min_confidence: float = 0.4,
        improve_accuracy: bool = False,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Run OCR on file bytes and return predictions with bounding boxes.

        Args:
            image_bytes: raw file bytes. Despite the name, may be any format
                the engine supports (images, PDFs, office docs, etc.).
            min_confidence: drop predictions below this score (0.0–1.0).
            improve_accuracy: engine-specific HQ mode toggle. Engines that
                don't support it should ignore the flag.
            filename: original filename, used to detect the file format /
                mime type. None when the input came in as a base64 string.

        Returns:
            {"predictions": [{"confidence": float, "label": str,
                              "x_min": int, "y_min": int,
                              "x_max": int, "y_max": int}, ...]}
            For non-image inputs that don't carry per-region geometry, bbox
            fields are all 0.
        """

    def warmup(self) -> None:
        """Optional eager initialization (model loading, etc.)."""
