from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..core.logging import get_logger
from .config import AIConfig
from .exceptions import AIModelError, AIUnavailableError
from .models import LaneSegmentationUNet

logger = get_logger("lanevision.ai.model_loader")


@dataclass(slots=True)
class ModelMetadata:
    model_name: str
    architecture: str
    device: str
    checkpoint_loaded: bool
    input_width: int
    input_height: int
    threshold: float
    parameter_count: int


class ModelLoader:
    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig.from_settings()
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._metadata: ModelMetadata | None = None
        self._load_error: str | None = None
        self._load_started_at: float | None = None
        self._load_count = 0
        self._lock = threading.Lock()
        self._failure: AIModelError | None = None

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self.load()
        return self._model

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            self.load()
        return self._metadata

    @property
    def load_count(self) -> int:
        """How many times the checkpoint file has been read (at most once per loader)."""
        return self._load_count

    @property
    def is_available(self) -> bool:
        if not self.config.enabled:
            return False
        if not self.config.model_path:
            return False
        if self._load_error is not None:
            return False
        return self._model is not None and self._metadata is not None

    def load(self) -> ModelMetadata:
        """Load the checkpoint once. Thread-safe; a failed load is remembered and reported again instead of retried,
        so a missing or broken checkpoint costs nothing per frame and every caller sees the same clear error."""
        if self._metadata is not None and self._model is not None:
            return self._metadata
        with self._lock:
            if self._metadata is not None and self._model is not None:
                return self._metadata
            if self._failure is not None:
                raise type(self._failure)(str(self._failure))
            try:
                return self._load_locked()
            except AIModelError as exc:
                self._failure = exc
                self._load_error = str(exc)
                raise

    def _load_locked(self) -> ModelMetadata:
        if not self.config.enabled:
            raise AIUnavailableError("AI mode is disabled")
        if not self.config.model_path:
            raise AIUnavailableError("AI model weights are not configured")
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise AIUnavailableError("AI model weights were not found")

        self._load_started_at = time.perf_counter()
        self._load_count += 1
        try:
            device = self._resolve_device()
            self._device = device
            model = LaneSegmentationUNet(in_channels=3, num_classes=2)
            state = torch.load(str(model_path), map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state_dict = state["state_dict"]
            elif isinstance(state, dict) and any(k.startswith("model") or k.startswith("backbone") for k in state):
                state_dict = state
            else:
                state_dict = state

            if isinstance(state_dict, dict):
                compatible_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("module."):
                        compatible_state_dict[key[7:]] = value
                    else:
                        compatible_state_dict[key] = value
                state_dict = compatible_state_dict

            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                raise AIModelError(f"Checkpoint is missing expected keys: {incompatible.missing_keys}")
            model.to(device)
            model.eval()
            self._model = model
            self._metadata = ModelMetadata(
                model_name=self.config.model_name,
                architecture=self.config.architecture,
                device=str(device),
                checkpoint_loaded=True,
                input_width=self.config.input_width,
                input_height=self.config.input_height,
                threshold=self.config.mask_threshold,
                parameter_count=sum(p.numel() for p in model.parameters()),
            )
            self._load_error = None
            logger.info("Loaded AI segmentation model %s on %s", self.config.model_name, device)
            return self._metadata
        except FileNotFoundError as exc:
            self._load_error = str(exc)
            raise AIUnavailableError("AI model weights were not found") from exc
        except RuntimeError as exc:
            self._load_error = str(exc)
            raise AIModelError(f"Unable to load AI checkpoint: {exc}") from exc
        except Exception as exc:
            self._load_error = str(exc)
            raise AIModelError(f"Unable to load AI checkpoint: {exc}") from exc

    def warmup(self) -> None:
        if not self.config.warmup or not self.is_available:
            return
        try:
            dummy = torch.zeros(1, 3, self.config.input_height, self.config.input_width, dtype=torch.float32, device=self._device or self._resolve_device())
            with torch.inference_mode():
                self.model(dummy)
        except Exception as exc:
            logger.warning("AI warmup failed: %s", exc)

    def _resolve_device(self) -> torch.device:
        requested = (self.config.device or "auto").strip().lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def get_status(self) -> dict[str, Any]:
        if not self.config.enabled:
            return {
                "enabled": False,
                "available": False,
                "model_name": self.config.model_name,
                "architecture": self.config.architecture,
                "device": self._resolve_device().type,
                "checkpoint_loaded": False,
                "input_size": [self.config.input_width, self.config.input_height],
                "error": "AI mode is disabled",
            }
        if not self.config.model_path:
            return {
                "enabled": True,
                "available": False,
                "model_name": self.config.model_name,
                "architecture": self.config.architecture,
                "device": self._resolve_device().type,
                "checkpoint_loaded": False,
                "input_size": [self.config.input_width, self.config.input_height],
                "error": "AI model checkpoint is not configured",
            }
        if self._load_error is not None:
            return {
                "enabled": True,
                "available": False,
                "model_name": self.config.model_name,
                "architecture": self.config.architecture,
                "device": self._resolve_device().type,
                "checkpoint_loaded": False,
                "input_size": [self.config.input_width, self.config.input_height],
                "error": self._load_error,
            }
        if self._model is None:
            return {
                "enabled": True,
                "available": False,
                "model_name": self.config.model_name,
                "architecture": self.config.architecture,
                "device": self._resolve_device().type,
                "checkpoint_loaded": False,
                "input_size": [self.config.input_width, self.config.input_height],
                "error": "Model not loaded",
            }
        return {
            "enabled": True,
            "available": True,
            "model_name": self.metadata.model_name,
            "architecture": self.metadata.architecture,
            "device": self.metadata.device,
            "checkpoint_loaded": self.metadata.checkpoint_loaded,
            "input_size": [self.config.input_width, self.config.input_height],
        }
