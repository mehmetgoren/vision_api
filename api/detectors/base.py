from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np


class Prediction(TypedDict):
    """CodeProject.AI-shaped per-object prediction."""

    confidence: float
    label: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class BaseDetector(ABC):
    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        min_confidence: float = 0.4,
    ) -> list[Prediction]:
        """Run detection on a BGR np.uint8 image; return CodeProject-shaped boxes."""

    def warmup(self) -> None:
        """Optional eager model load."""
