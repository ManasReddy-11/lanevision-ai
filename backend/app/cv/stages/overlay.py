from __future__ import annotations

import cv2
import numpy as np

from ..config import LaneDetectionConfig
from ..types import LaneLine, LaneDetectionResult


def draw_lane_overlay(frame: np.ndarray, left_lane: LaneLine | None, right_lane: LaneLine | None, config: LaneDetectionConfig | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Draw lane boundaries and an optional filled polygon on the frame."""
    config = config or LaneDetectionConfig()
    output = frame.copy()
    polygon: np.ndarray | None = None

    if left_lane is not None:
        left = left_lane.clip_to_bounds(output.shape[1], output.shape[0])
        cv2.line(output, (int(left.x1), int(left.y1)), (int(left.x2), int(left.y2)), (255, 0, 0), config.line_thickness)

    if right_lane is not None:
        right = right_lane.clip_to_bounds(output.shape[1], output.shape[0])
        cv2.line(output, (int(right.x1), int(right.y1)), (int(right.x2), int(right.y2)), (0, 255, 255), config.line_thickness)

    if left_lane is not None and right_lane is not None:
        left = left_lane.clip_to_bounds(output.shape[1], output.shape[0])
        right = right_lane.clip_to_bounds(output.shape[1], output.shape[0])
        polygon = np.array([[left.x1, left.y1], [right.x1, right.y1], [right.x2, right.y2], [left.x2, left.y2]], dtype=np.int32)
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], (0, 255, 0))
        output = cv2.addWeighted(output, 1 - config.overlay_opacity, overlay, config.overlay_opacity, 0)

    return output, polygon
