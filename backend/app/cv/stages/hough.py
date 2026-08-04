from __future__ import annotations

import cv2
import numpy as np

from ..config import LaneDetectionConfig
from ..types import LaneSegment


def detect_hough_lines(frame: np.ndarray, config: LaneDetectionConfig | None = None) -> list[LaneSegment]:
    """Detect lane-like line segments with Hough transform."""
    config = config or LaneDetectionConfig()
    if frame.size == 0:
        return []

    lines = cv2.HoughLinesP(
        frame,
        config.hough_rho,
        config.hough_theta,
        config.hough_threshold,
        minLineLength=config.hough_min_line_length,
        maxLineGap=config.hough_max_line_gap,
    )
    if lines is None:
        return []

    segments: list[LaneSegment] = []
    for line in lines:
        line_values = line[0] if len(line) > 0 else None
        if line_values is None:
            continue
        if isinstance(line_values, (list, tuple)) and len(line_values) >= 4:
            x1, y1, x2, y2 = line_values[:4]
        elif isinstance(line_values, np.ndarray) and line_values.size >= 4:
            x1, y1, x2, y2 = line_values[:4]
        else:
            continue

        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        if abs(x2 - x1) < 1e-6:
            continue
        segment = LaneSegment(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
        if segment.length is None:
            continue
        if segment.length < config.min_segment_length:
            continue
        segments.append(segment)
    return segments
