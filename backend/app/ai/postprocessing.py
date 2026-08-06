from __future__ import annotations

import cv2
import numpy as np

from .config import AIConfig


class MaskPostprocessor:
    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig.from_settings()

    def process(self, mask: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
        if mask is None:
            raise ValueError("Mask is required")
        if mask.ndim != 2:
            raise ValueError("Expected a 2D mask")
        if mask.size == 0:
            raise ValueError("Mask is empty")

        resized = cv2.resize(mask.astype(np.float32), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        resized = (resized >= self.config.mask_threshold).astype(np.uint8)

        if self.config.min_component_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(resized, connectivity=8)
            cleaned = np.zeros_like(resized)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= self.config.min_component_area:
                    cleaned[labels == label_id] = 1
            resized = cleaned

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        resized = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel)
        return resized
