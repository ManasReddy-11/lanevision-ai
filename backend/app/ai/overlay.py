from __future__ import annotations

import cv2
import numpy as np

from .config import AIConfig


class OverlayRenderer:
    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig.from_settings()

    def render(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("Frame is required")
        if mask is None:
            raise ValueError("Mask is required")
        overlay = frame.copy()
        color_mask = np.zeros_like(overlay)
        color_mask[:, :, 1] = (mask * 255).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 1.0 - self.config.overlay_alpha, color_mask, self.config.overlay_alpha, 0)
        return overlay
