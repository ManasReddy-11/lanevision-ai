from __future__ import annotations

import numpy as np


def sliding_window_search(binary: np.ndarray, window_count: int = 9, margin: int = 70) -> tuple[list[int], list[int], list[int], list[int]]:
    """Perform a simple sliding-window search and return pixel indices for left/right lanes."""
    if binary.size == 0:
        return [], [], [], []
    height, width = binary.shape
    window_height = max(1, height // max(window_count, 1))
    non_zero_y, non_zero_x = np.where(binary > 0)
    if len(non_zero_x) == 0:
        return [], [], [], []

    left_x: list[int] = []
    left_y: list[int] = []
    right_x: list[int] = []
    right_y: list[int] = []

    left_current = width // 4
    right_current = width * 3 // 4

    for window_idx in range(window_count):
        y_low = height - (window_idx + 1) * window_height
        y_high = height - window_idx * window_height
        if y_low < 0:
            y_low = 0
        if y_high > height:
            y_high = height
        window_mask = (non_zero_y >= y_low) & (non_zero_y < y_high)
        if not np.any(window_mask):
            continue
        window_x = non_zero_x[window_mask]
        window_y = non_zero_y[window_mask]

        left_window = (window_x >= left_current - margin) & (window_x <= left_current + margin)
        right_window = (window_x >= right_current - margin) & (window_x <= right_current + margin)
        if np.any(left_window):
            left_x.extend(int(x) for x in window_x[left_window])
            left_y.extend(int(y) for y in window_y[left_window])
            left_current = int(np.mean(window_x[left_window]))
        if np.any(right_window):
            right_x.extend(int(x) for x in window_x[right_window])
            right_y.extend(int(y) for y in window_y[right_window])
            right_current = int(np.mean(window_x[right_window]))

    return left_x, left_y, right_x, right_y
