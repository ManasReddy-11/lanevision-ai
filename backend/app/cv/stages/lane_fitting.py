from __future__ import annotations

import numpy as np

from ..config import LaneDetectionConfig
from ..types import LaneLine, LaneSegment


def calculate_slope(segment: LaneSegment) -> float:
    """Calculate slope for a segment if safe to do so."""
    if abs(segment.x2 - segment.x1) < 1e-6:
        return 0.0
    return (segment.y2 - segment.y1) / (segment.x2 - segment.x1)


def is_horizontal_line(segment: LaneSegment, config: LaneDetectionConfig | None = None) -> bool:
    """Reject almost-horizontal line segments."""
    config = config or LaneDetectionConfig()
    slope = abs(calculate_slope(segment))
    return slope < 1e-3 or slope < config.min_slope_abs


def segment_length(segment: LaneSegment) -> float:
    """Calculate the Euclidean length of a segment."""
    dx = segment.x2 - segment.x1
    dy = segment.y2 - segment.y1
    return float((dx * dx + dy * dy) ** 0.5)


def classify_lane_segments(left_segments: list[LaneSegment], right_segments: list[LaneSegment], width: int, height: int) -> tuple[LaneSegment | None, LaneSegment | None]:
    """Classify candidate segments into left and right lane groups."""
    center_x = width / 2.0

    def _classify(segments: list[LaneSegment], prefer_left: bool) -> LaneSegment | None:
        if not segments:
            return None
        if prefer_left:
            candidates = [segment for segment in segments if (segment.x1 + segment.x2) / 2.0 < center_x]
        else:
            candidates = [segment for segment in segments if (segment.x1 + segment.x2) / 2.0 > center_x]
        if not candidates:
            candidates = segments
        candidates.sort(key=lambda item: (item.length or 0.0), reverse=True)
        return candidates[0]

    left = _classify(left_segments, True)
    right = _classify(right_segments, False)
    return left, right


def fit_lane_line(segments: list[LaneSegment], width: int, height: int, roi_top_y: int, roi_bottom_y: int) -> LaneLine | None:
    """Fit a stable lane line from multiple supporting segments."""
    if not segments:
        return None

    valid_segments = [segment for segment in segments if segment.length is not None and segment.length >= 1.0]
    if not valid_segments:
        return None

    weighted = sorted(valid_segments, key=lambda item: item.length or 0.0, reverse=True)
    slope_values = [calculate_slope(segment) for segment in weighted[:5]]
    average_slope = float(np.mean(slope_values)) if slope_values else 0.0

    top_y = max(roi_top_y, 0)
    bottom_y = min(roi_bottom_y, height - 1)
    if bottom_y <= top_y:
        bottom_y = height - 1

    dominant_segment = weighted[0]
    x1 = dominant_segment.x1
    y1 = dominant_segment.y1
    x2 = dominant_segment.x2
    y2 = dominant_segment.y2

    if abs(x2 - x1) < 1e-6:
        x2 = x1 + 1.0

    return LaneLine(
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        slope=average_slope,
        confidence=min(1.0, max(0.0, (dominant_segment.length or 0.0) / max(width, 1))),
        support_segments=weighted[:5],
    )
