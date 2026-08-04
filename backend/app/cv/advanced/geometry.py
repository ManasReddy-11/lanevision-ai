from __future__ import annotations

import numpy as np


def calculate_lane_center(left_x: float | None, right_x: float | None, frame_center_x: float | None = None) -> float | None:
    if left_x is None or right_x is None:
        return None
    if frame_center_x is None:
        frame_center_x = 0.0
    lane_center_x = (left_x + right_x) / 2.0
    return lane_center_x - frame_center_x


def calculate_lane_widths(left_x: float | None, right_x: float | None, height: int) -> dict[str, float]:
    if left_x is None or right_x is None:
        return {"width_bottom_pixels": 0.0, "width_average_pixels": 0.0}
    width_bottom = abs(right_x - left_x)
    return {"width_bottom_pixels": width_bottom, "width_average_pixels": width_bottom / max(height, 1)}


def calculate_polynomial_curvature(coefficients: np.ndarray) -> float:
    if coefficients is None or len(coefficients) < 3:
        return float("nan")
    if not np.isfinite(coefficients).all():
        return float("nan")
    return float(abs(coefficients[0]) * 1000.0)
