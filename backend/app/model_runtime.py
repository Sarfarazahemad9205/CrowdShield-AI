"""Shared YOLO runtime with automatic device selection and isolated video runs."""

from functools import wraps
from pathlib import Path
from threading import RLock

import torch
from ultralytics import YOLO


class YoloRuntime:
    """Lazily load one model and serialize access to its persistent tracker state."""

    def __init__(self, weights_path: Path | None = None) -> None:
        self.weights_path = weights_path or Path(__file__).resolve().parents[1] / "yolo11n.pt"
        self._model: YOLO | None = None
        self._lock = RLock()

    @property
    def device(self) -> str:
        return "0" if torch.cuda.is_available() else "cpu"

    @property
    def use_half_precision(self) -> bool:
        return torch.cuda.is_available()

    def _get_model(self) -> YOLO:
        if self._model is None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")
            self._model = YOLO(str(self.weights_path))
        return self._model

    @property
    def model(self) -> YOLO:
        """Return the shared, lazily initialized model."""

        return self._get_model()

    def serialized(self, function):
        """Serialize complete video runs because YOLO retains tracker state."""

        @wraps(function)
        def run(*args, **kwargs):
            with self._lock:
                model = self._get_model()
                self._reset_trackers(model)
                try:
                    return function(*args, **kwargs)
                finally:
                    self._reset_trackers(model)

        return run

    @staticmethod
    def _reset_trackers(model: YOLO) -> None:
        """Prevent persistent IDs/state from leaking between uploaded videos."""

        predictor = getattr(model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

yolo_runtime = YoloRuntime()
