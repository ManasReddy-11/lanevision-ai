from __future__ import annotations

import cv2
import numpy as np

from ..config import LaneDetectionConfig


def get_roi_vertices(width: int, height: int, config: LaneDetectionConfig | None = None) -> list[tuple[float, float]]:
    """Pixel coordinates of the trapezoidal ROI, in the order of config.roi_polygon (bottom-left, top-left, top-right,
    bottom-right), which traces the trapezoid's outline without crossing itself."""
    config = config or LaneDetectionConfig()
    return [(int(round(width * x)), int(round(height * y))) for x, y in config.roi_polygon]


def get_roi_vertices_ordered(width: int, height: int, config: LaneDetectionConfig | None = None) -> list[tuple[float, float]]:
    """Return ROI vertices in a consistent order for polygon drawing."""
    return get_roi_vertices(width, height, config)


def create_roi_mask(width: int, height: int, config: LaneDetectionConfig | None = None) -> np.ndarray:
    """Create a binary ROI mask matching the input image size."""
    mask = np.zeros((height, width), dtype=np.uint8)
    vertices = np.array(get_roi_vertices_ordered(width, height, config), dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 255)
    return mask


def apply_roi(frame: np.ndarray, config: LaneDetectionConfig | None = None) -> np.ndarray:
    """Apply an ROI mask to an edge image."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame.shape[:2]
    mask = create_roi_mask(width, height, config)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked
