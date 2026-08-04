from __future__ import annotations

import numpy as np


class CVException(ValueError):
    """Raised when lane detection input is invalid."""


def validate_frame(frame: np.ndarray | None) -> np.ndarray:
    """Validate that the input frame is a usable NumPy array."""
    if frame is None:
        raise CVException("Frame cannot be None")

    array = np.asarray(frame)
    if not isinstance(array, np.ndarray):
        raise CVException("Frame must be a NumPy array")

    if array.ndim not in {2, 3}:
        raise CVException("Frame must be a 2D grayscale image or 3D color image")

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise CVException("Frame width and height must be non-zero")

    if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
        raise CVException("Frame must contain 1, 3, or 4 channels")

    if array.size == 0:
        raise CVException("Frame array is empty")

    return array
