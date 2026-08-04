from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class PolynomialLane:
    coefficients: np.ndarray
    pixel_count: int
    fit_x: np.ndarray
    plot_y: np.ndarray
    confidence: float


def fit_polynomial_lane(x: np.ndarray, y: np.ndarray, min_pixels: int = 10) -> PolynomialLane:
    """Fit a second-order polynomial in x = Ay^2 + By + C."""
    if len(x) < min_pixels:
        return PolynomialLane(coefficients=np.array([np.nan, np.nan, np.nan], dtype=np.float32), pixel_count=int(len(x)), fit_x=np.array([], dtype=np.float32), plot_y=np.array([], dtype=np.float32), confidence=0.0)

    y_arr = np.asarray(y, dtype=np.float32)
    x_arr = np.asarray(x, dtype=np.float32)
    if y_arr.size < 2:
        return PolynomialLane(coefficients=np.array([np.nan, np.nan, np.nan], dtype=np.float32), pixel_count=int(len(x)), fit_x=np.array([], dtype=np.float32), plot_y=np.array([], dtype=np.float32), confidence=0.0)

    try:
        coefficients = np.polyfit(y_arr, x_arr, 2)
    except np.linalg.LinAlgError:
        coefficients = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    plot_y = np.linspace(int(np.min(y_arr)), int(np.max(y_arr)), 20)
    fit_x = np.polyval(coefficients, plot_y)
    return PolynomialLane(coefficients=coefficients.astype(np.float32), pixel_count=int(len(x)), fit_x=fit_x.astype(np.float32), plot_y=plot_y.astype(np.float32), confidence=0.5)


def validate_polynomial_lane(lane: PolynomialLane, width: int, height: int) -> bool:
    if lane is None:
        return False
    if not np.isfinite(lane.coefficients).all():
        return False
    if lane.pixel_count <= 0:
        return False
    if lane.fit_x.size == 0 or lane.plot_y.size == 0:
        return False
    if np.any(lane.fit_x < -50) or np.any(lane.fit_x > width + 50):
        return False
    return True
