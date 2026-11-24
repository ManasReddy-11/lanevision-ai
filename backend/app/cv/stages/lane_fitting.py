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
    """Fit one lane boundary to its supporting segments and extend it over the ROI rows.

    The line x = m * y + c is a length-weighted least-squares fit through the endpoints of the (up to five) longest
    segments. The result runs from the ROI top (x1, y1) to the ROI bottom (x2, y2), so x2 is the boundary position
    nearest the vehicle whichever order the Hough transform returned the endpoints in.
    """
    if not segments:
        return None

    valid_segments = [segment for segment in segments if segment.length is not None and segment.length >= 1.0]
    if not valid_segments:
        return None

    weighted = sorted(valid_segments, key=lambda item: item.length or 0.0, reverse=True)[:5]
    ys = np.array([value for segment in weighted for value in (segment.y1, segment.y2)], dtype=np.float64)
    xs = np.array([value for segment in weighted for value in (segment.x1, segment.x2)], dtype=np.float64)
    weights = np.repeat([float(segment.length or 0.0) for segment in weighted], 2)
    if np.ptp(ys) < 1e-6:
        return None  # horizontal support only: no lane direction to extend
    m, c = np.polyfit(ys, xs, 1, w=np.sqrt(weights))

    top_y = float(max(roi_top_y, 0))
    bottom_y = float(min(roi_bottom_y, height - 1))
    if bottom_y <= top_y:
        top_y, bottom_y = 0.0, float(height - 1)
    x_top, x_bottom = m * top_y + c, m * bottom_y + c
    slope = (bottom_y - top_y) / (x_bottom - x_top) if abs(x_bottom - x_top) > 1e-6 else 0.0

    return LaneLine(
        x1=float(x_top),
        y1=top_y,
        x2=float(x_bottom),
        y2=bottom_y,
        slope=float(slope),
        confidence=min(1.0, max(0.0, (weighted[0].length or 0.0) / max(width, 1))),
        support_segments=weighted,
    )
