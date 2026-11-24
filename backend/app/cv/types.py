from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(slots=True)
class LaneSegment:
    """A detected line segment from the Hough transform stage."""

    x1: float
    y1: float
    x2: float
    y2: float
    slope: Optional[float] = None
    length: Optional[float] = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.slope is None:
            self.slope = self._calculate_slope()
        if self.length is None:
            self.length = self._calculate_length()

    def _calculate_slope(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        if abs(dx) < 1e-6:
            return 0.0
        return dy / dx

    def _calculate_length(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return float((dx * dx + dy * dy) ** 0.5)


@dataclass(slots=True)
class LaneLine:
    """A fitted lane boundary represented by two endpoints."""

    x1: float
    y1: float
    x2: float
    y2: float
    slope: float
    confidence: float
    support_segments: list[LaneSegment] = field(default_factory=list)

    def clip_to_bounds(self, width: int, height: int) -> "LaneLine":
        """Clamp the lane line endpoints to image bounds."""
        x1 = max(0, min(int(round(self.x1)), width - 1))
        y1 = max(0, min(int(round(self.y1)), height - 1))
        x2 = max(0, min(int(round(self.x2)), width - 1))
        y2 = max(0, min(int(round(self.y2)), height - 1))
        return LaneLine(x1=x1, y1=y1, x2=x2, y2=y2, slope=self.slope, confidence=self.confidence, support_segments=list(self.support_segments))


@dataclass(slots=True)
class LaneDetectionResult:
    """Structured output from the lane detection pipeline."""

    success: bool
    processed_frame: np.ndarray
    left_lane: Optional[LaneLine]
    right_lane: Optional[LaneLine]
    lane_polygon: Optional[np.ndarray]
    lane_center_x: Optional[float]
    frame_center_x: Optional[float]
    vehicle_offset_pixels: Optional[float]
    vehicle_offset_normalized: Optional[float]
    confidence: float
    processing_time_ms: float
    detected_segment_count: int
    lane_width_bottom_pixels: Optional[float] = None
    lane_width_average_pixels: Optional[float] = None
    left_curvature_pixels: Optional[float] = None
    right_curvature_pixels: Optional[float] = None
    average_curvature_pixels: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    debug_output: Optional[dict[str, np.ndarray]] = None
