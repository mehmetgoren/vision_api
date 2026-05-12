"""CodeProject.AI-compatible response envelope.

Every CodeProject.AI vision endpoint returns the same shape: success/message/
predictions plus timing and module-identification fields. Clients (Blue Iris,
Home Assistant, Frigate add-ons) parse these field names exactly, so the
helpers in this module emit them verbatim.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class Timing:
    """Tracks inference and total processing time in milliseconds.

    Usage:
        with Timing() as t:
            with t.inference():
                model.predict(...)
            ...post-processing...
        # t.process_ms, t.inference_ms are now populated.
    """

    def __init__(self) -> None:
        self.process_ms: int = 0
        self.inference_ms: int = 0
        self._t0: float = 0.0
        self._inf_t0: float = 0.0

    def __enter__(self) -> "Timing":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.process_ms = int((time.perf_counter() - self._t0) * 1000)

    @contextmanager
    def inference(self) -> Iterator[None]:
        self._inf_t0 = time.perf_counter()
        try:
            yield
        finally:
            self.inference_ms = int((time.perf_counter() - self._inf_t0) * 1000)


def cuda_available() -> bool:
    """Cheap probe: is CUDA usable in this process?

    Tried providers, in order: torch.cuda, onnxruntime CUDAExecutionProvider.
    Returns False on any failure so the helper is safe to call before any
    backend has been loaded.
    """
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return True
    except Exception:
        pass
    return False


def execution_provider(forced: str | None = None) -> str:
    """Return "CUDA" or "CPU" — the engine itself may override via `forced`."""
    if forced:
        return forced
    return "CUDA" if cuda_available() else "CPU"


def build_response(
    predictions: list[dict[str, Any]] | None,
    *,
    module_id: str,
    module_name: str,
    command: str,
    timing: Timing,
    execution_provider_name: str | None = None,
    can_use_gpu: bool | None = None,
    extra: dict[str, Any] | None = None,
    message: str = "",
) -> dict[str, Any]:
    """Assemble the standard CodeProject.AI success envelope."""
    provider = execution_provider(execution_provider_name)
    use_gpu = cuda_available() if can_use_gpu is None else can_use_gpu

    body: dict[str, Any] = {
        "success": True,
        "message": message,
        "predictions": predictions if predictions is not None else [],
        "count": len(predictions) if predictions is not None else 0,
        "inferenceMs": timing.inference_ms,
        "processMs": timing.process_ms,
        "moduleId": module_id,
        "moduleName": module_name,
        "command": command,
        "executionProvider": provider,
        "canUseGPU": use_gpu,
        "analysisRoundTripMs": timing.process_ms,
    }
    if extra:
        body.update(extra)
    return body


def error_response(
    message: str,
    *,
    module_id: str,
    command: str,
    timing: Timing | None = None,
) -> dict[str, Any]:
    """Standard error envelope. HTTP 200 — matches CodeProject's convention of
    surfacing inference errors in-band rather than via 4xx/5xx.
    """
    ms = timing.process_ms if timing else 0
    return {
        "success": False,
        "message": message,
        "error": message,
        "predictions": [],
        "count": 0,
        "inferenceMs": 0,
        "processMs": ms,
        "moduleId": module_id,
        "moduleName": module_id,
        "command": command,
        "executionProvider": execution_provider(),
        "canUseGPU": cuda_available(),
        "analysisRoundTripMs": ms,
    }
