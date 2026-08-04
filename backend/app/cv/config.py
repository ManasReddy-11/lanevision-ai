from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(slots=True)
class LaneDetectionConfig:
    """Configuration for the classical lane detection pipeline."""

    blur_kernel_size: int = 5
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150

    white_thresholds: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((200, 255, 200), (255, 255, 255))
    yellow_thresholds: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((15, 80, 80), (45, 255, 255))

    roi_polygon: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.10, 0.95),
        (0.40, 0.60),
        (0.60, 0.60),
        (0.90, 0.95),
    )

    hough_rho: float = 1.0
    hough_theta: float = 3.141592653589793 / 180.0
    hough_threshold: int = 20
    hough_min_line_length: int = 30
    hough_max_line_gap: int = 10

    min_slope_abs: float = 0.2
    max_slope_abs: float = 1.5
    min_segment_length: float = 20.0

    line_thickness: int = 5
    overlay_opacity: float = 0.35

    lane_smoothing_alpha: float = 0.4

    debug: bool = False
    min_supported_segments: int = 2
    max_num_segments: int = 20
    roi_margin: float = 0.02

    resize_width: int = 1280
    resize_height: int = 720
    resize_enabled: bool = True

    markers: Tuple[str, ...] = field(default_factory=lambda: ("white", "yellow"))
