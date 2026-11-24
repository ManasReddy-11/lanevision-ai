from __future__ import annotations

import cv2
import numpy as np
import torch

from .config import AIConfig


class Preprocessor:
    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig.from_settings()

    def prepare(self, frame: np.ndarray, device: torch.device, size: tuple[int, int] | None = None) -> tuple[torch.Tensor, tuple[int, int]]:
        """size: model input (width, height); defaults to the configured input size."""
        if frame is None:
            raise ValueError("Input frame is required")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR image with 3 channels")
        if not np.isfinite(frame).all():
            raise ValueError("Input frame contains invalid values")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width, height = size or (self.config.input_width, self.config.input_height)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        return tensor, (frame.shape[1], frame.shape[0])
