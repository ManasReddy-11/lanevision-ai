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

    # OpenCV 4 returns an (N, 1, 4) array and OpenCV 5 an (N, 4) array; both are (x1, y1, x2, y2) per segment.
    segments: list[LaneSegment] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4).tolist():
        if abs(x2 - x1) < 1e-6:
            continue
        segment = LaneSegment(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
        if segment.length is None or segment.length < config.min_segment_length:
            continue
        segments.append(segment)
    return segments
