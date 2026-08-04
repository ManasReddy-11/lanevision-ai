from __future__ import annotations

import cv2
import numpy as np

from ..config import LaneDetectionConfig
from ..utils.validation import validate_frame


def resize_if_required(frame: np.ndarray, config: LaneDetectionConfig | None = None) -> np.ndarray:
    """Resize the frame if the configuration enables it and the frame is larger."""
    validated = validate_frame(frame)
    config = config or LaneDetectionConfig()

    if not config.resize_enabled:
        return validated

    height, width = validated.shape[:2]
    if width <= config.resize_width and height <= config.resize_height:
        return validated

    scale = min(config.resize_width / max(width, 1), config.resize_height / max(height, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(validated, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert a color frame to grayscale without mutating the input."""
    validated = validate_frame(frame)
    if validated.ndim == 2:
        return validated.copy()
    gray = cv2.cvtColor(validated, cv2.COLOR_BGR2GRAY)
    return gray


def to_hls(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to HLS color space."""
    validated = validate_frame(frame)
    if validated.ndim == 2:
        return cv2.cvtColor(cv2.cvtColor(validated, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HLS)
    return cv2.cvtColor(validated, cv2.COLOR_BGR2HLS)


def gaussian_blur(frame: np.ndarray, config: LaneDetectionConfig | None = None) -> np.ndarray:
    """Apply Gaussian blur to the frame."""
    validated = validate_frame(frame)
    config = config or LaneDetectionConfig()
    kernel_size = config.blur_kernel_size if config.blur_kernel_size % 2 == 1 else config.blur_kernel_size + 1
    return cv2.GaussianBlur(validated, (kernel_size, kernel_size), 0)
