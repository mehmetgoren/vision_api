from api.ocrs.base import BaseOcr
from api.ocrs.kreuzberg_ocr import KreuzbergOcr, kreuzberg_ocr
from api.ocrs.paddle_ocr import PaddleOcr, paddle_ocr

__all__ = [
    "BaseOcr",
    "KreuzbergOcr",
    "PaddleOcr",
    "kreuzberg_ocr",
    "paddle_ocr",
]
