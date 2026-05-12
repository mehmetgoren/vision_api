"""Places365 (ResNet50) scene classifier.

Mirrors CodeProject.AI's scene module contract: returns top-1 (label,
confidence) over the 365 Places categories. Weights and label list are
downloaded on first use from the official MIT mirror and cached to
MODELS_DIR/places365.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)


_TORCH_MISSING_DETAIL = (
    "PyTorch/torchvision not installed. Run `uv sync` to pull `torch` and "
    "`torchvision` from the cu126 index."
)

_WEIGHTS_URL = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
_LABELS_URL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"


def _places365_dir() -> Path:
    base = Path(
        os.environ.get("PLACES365_WEIGHTS_DIR")
        or (Path(os.environ.get("MODELS_DIR", "models")) / "places365")
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    logger.info("Downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


class Places365Classifier:
    def __init__(self) -> None:
        self._model: Any = None
        self._labels: list[str] | None = None
        self._device: str = "cpu"
        self._transform: Any = None

    def _load_labels(self) -> list[str]:
        if self._labels is not None:
            return self._labels
        labels_path = _places365_dir() / "categories_places365.txt"
        _download(_LABELS_URL, labels_path)
        labels: list[str] = []
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: "/a/airfield 0"
            name = line.split()[0]
            if name.startswith("/"):
                name = name[3:]  # drop "/a/" prefix
            labels.append(name)
        if len(labels) != 365:
            logger.warning("Places365 labels file had %d entries (expected 365)", len(labels))
        self._labels = labels
        return labels

    def _get_model(self) -> tuple[Any, str, Any]:
        if self._model is not None and self._transform is not None:
            return self._model, self._device, self._transform

        try:
            import torch  # type: ignore[import-not-found]
            from torchvision import transforms  # type: ignore[import-not-found]
            from torchvision.models import resnet50  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning("torch/torchvision not installed: %s", exc)
            raise HTTPException(status_code=503, detail=_TORCH_MISSING_DETAIL) from exc

        weights_path = _places365_dir() / "resnet50_places365.pth.tar"
        _download(_WEIGHTS_URL, weights_path)

        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 365)
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        # Checkpoint key shape varies between Caffe-converted and PyTorch-native dumps;
        # try both common layouts.
        if isinstance(state, dict) and "state_dict" in state:
            raw = state["state_dict"]
        else:
            raw = state
        cleaned = {k.replace("module.", ""): v for k, v in raw.items()}
        model.load_state_dict(cleaned, strict=False)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._model = model
        self._device = device
        self._transform = transform
        logger.info("Places365 ResNet50 loaded (device=%s, weights=%s)", device, weights_path)
        return model, device, transform

    def warmup(self) -> None:
        _ = self._load_labels()
        _ = self._get_model()

    def classify(self, image_bgr: np.ndarray, top_k: int = 5) -> dict[str, Any]:
        import torch  # type: ignore[import-not-found]

        labels = self._load_labels()
        model, device, transform = self._get_model()

        rgb = image_bgr[:, :, ::-1].copy()  # BGR -> RGB
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = np.argsort(probs)[::-1][:top_k]
        top = [
            {"label": labels[int(i)] if int(i) < len(labels) else str(int(i)),
             "confidence": float(probs[int(i)])}
            for i in top_idx
        ]
        return {
            "label": top[0]["label"],
            "confidence": round(top[0]["confidence"], 4),
            "top_k": [{"label": t["label"], "confidence": round(t["confidence"], 4)} for t in top],
        }


scene_classifier = Places365Classifier()
