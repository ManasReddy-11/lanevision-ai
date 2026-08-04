from __future__ import annotations

import cv2
import numpy as np

from ..config import LaneDetectionConfig
from ..utils.validation import validate_frame


def detect_edges(frame: np.ndarray, config: LaneDetectionConfig | None = None) -> np.ndarray:
    """Run Canny edge detection on a filtered grayscale frame."""
    validated = validate_frame(frame)
    config = config or LaneDetectionConfig()

    if validated.ndim == 3:
        gray = cv2.cvtColor(validated, cv2.COLOR_BGR2GRAY)
    else:
        gray = validated

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(
        blurred,
        config.canny_low_threshold,
        config.canny_high_threshold,
    )
    return edges
