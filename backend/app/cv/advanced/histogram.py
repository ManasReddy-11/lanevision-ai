from __future__ import annotations

import numpy as np


def compute_histogram_lane_bases(binary: np.ndarray, split_ratio: float = 0.5) -> tuple[int, int]:
    """Return the likely left and right base positions from the lower histogram."""
    if binary.size == 0:
        return 0, 0
    histogram = np.sum(binary[binary.shape[0] // 2 :, :], axis=0)
    if histogram.size == 0:
        return 0, 0
    midpoint = int(round(split_ratio * binary.shape[1]))
    left_indices = np.argmax(histogram[:midpoint]) if midpoint > 0 else 0
    right_indices = midpoint + np.argmax(histogram[midpoint:]) if midpoint < len(histogram) else len(histogram) - 1
    return int(left_indices), int(right_indices)
