from __future__ import annotations

import time

import numpy as np
import torch

from .config import AIConfig
from .exceptions import AIModelError
from .model_loader import ModelLoader
from .preprocessing import Preprocessor


class AIInferenceEngine:
    def __init__(self, config: AIConfig | None = None, loader: ModelLoader | None = None) -> None:
        self.config = config or AIConfig.from_settings()
        self.loader = loader or ModelLoader(self.config)
        self.preprocessor = Preprocessor(self.config)

    def run(self, frame: np.ndarray) -> dict[str, object]:
        if frame is None:
            raise AIModelError("Input frame is required")
        model = self.loader.model
        device = self.loader._device or torch.device("cpu")
        start = time.perf_counter()
        tensor, original_size = self.preprocessor.prepare(frame, device)
        with torch.inference_mode():
            output = model(tensor)
        inference_time_ms = (time.perf_counter() - start) * 1000.0
        logits = output["out"] if isinstance(output, dict) and "out" in output else output[0]
        if isinstance(logits, torch.Tensor):
            logits = logits.squeeze(0)
            logits = logits.detach().cpu()
            probs = torch.softmax(logits, dim=0)
            lane_prob = probs[self.config.lane_class_id].numpy()
            mask = lane_prob >= self.config.mask_threshold
        else:
            raise AIModelError("Model output was not a tensor")
        return {
            "logits": logits.numpy(),
            "probability_mask": lane_prob,
            "mask": mask.astype(np.uint8),
            "original_size": original_size,
            "inference_time_ms": inference_time_ms,
            "model_metadata": self.loader.metadata,
        }
