from __future__ import annotations

import cv2
import numpy as np

from ..config import LaneDetectionConfig
from ..utils.validation import validate_frame


def apply_lane_color_filter(frame: np.ndarray, config: LaneDetectionConfig | None = None) -> np.ndarray:
    """Create a binary mask for white and yellow lane markings."""
    validated = validate_frame(frame)
    config = config or LaneDetectionConfig()

    if validated.ndim == 2:
        gray = validated
        mask = np.zeros(gray.shape, dtype=np.uint8)
    else:
        hls = cv2.cvtColor(validated, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)

        white_lower, white_upper = config.white_thresholds
        yellow_lower, yellow_upper = config.yellow_thresholds

        white_mask = np.logical_and.reduce(
            (
                l >= white_lower[0],
                l <= white_upper[0],
                s >= white_lower[1],
                s <= white_upper[1],
                h >= white_lower[2],
                h <= white_upper[2],
            )
        )

        yellow_mask = np.logical_and.reduce(
            (
                h >= yellow_lower[0],
                h <= yellow_upper[0],
                l >= yellow_lower[1],
                l <= yellow_upper[1],
                s >= yellow_lower[2],
                s <= yellow_upper[2],
            )
        )

        mask = np.zeros_like(l, dtype=np.uint8)
        mask[white_mask | yellow_mask] = 255
        return mask

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[gray > 180] = 255
    return mask
