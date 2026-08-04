from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(slots=True)
class AdvancedLaneDetectionConfig:
    """Configuration for the advanced curved-lane detector."""

    perspective_src_points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.20, 0.95),
        (0.35, 0.62),
        (0.65, 0.62),
        (0.80, 0.95),
    )
    perspective_dst_points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.20, 1.00),
        (0.20, 0.00),
        (0.80, 0.00),
        (0.80, 1.00),
    )

    sobel_kernel_size: int = 3
    threshold_low: int = 50
    threshold_high: int = 150
    white_thresholds: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((200, 255, 200), (255, 255, 255))
    yellow_thresholds: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((15, 80, 80), (45, 255, 255))

    sliding_window_count: int = 9
    window_margin: int = 70
    min_pixels_recenter: int = 25
    min_pixels_fit: int = 10
    max_poly_change: float = 200.0
    lane_width_constraint: float = 200.0
    curvature_sanity_threshold: float = 5000.0
    overlay_opacity: float = 0.35
    search_margin: int = 40
    debug: bool = False

    lane_width_bottom_pixels: Tuple[float, float] = (0.0, 0.0)
    lane_width_average_pixels: Tuple[float, float] = (0.0, 0.0)

    use_assumed_physical_scaling: bool = False
    ym_per_pix: float = 30.0 / 720.0
    xm_per_pix: float = 3.7 / 700.0

    markers: Tuple[str, ...] = field(default_factory=lambda: ("white", "yellow"))
