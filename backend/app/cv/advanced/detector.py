from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from ..config import LaneDetectionConfig
from ..detector import ClassicLaneDetector
from ..types import LaneDetectionResult, LaneLine
from ..utils.validation import CVException, validate_frame
from .config import AdvancedLaneDetectionConfig
from ..metrics.lane_metrics import calculate_lane_center, calculate_vehicle_offset, calculate_vehicle_offset_normalized, reference_row
from .geometry import calculate_polynomial_curvature
from .histogram import compute_histogram_lane_bases
from .perspective import get_perspective_matrices, unwarp_perspective, warp_perspective
from .polynomial import PolynomialLane, fit_polynomial_lane, validate_polynomial_lane
from .sliding_windows import sliding_window_search
from .temporal import AdvancedTemporalState
from .thresholding import create_lane_binary


class AdvancedLaneDetector:
    """A simple advanced curved-lane detector built with classical computer vision."""

    def __init__(self, config: AdvancedLaneDetectionConfig | None = None) -> None:
        self.config = config or AdvancedLaneDetectionConfig()
        self.temporal_state = AdvancedTemporalState()

    def detect(self, frame: np.ndarray, debug: bool = False) -> LaneDetectionResult:
        started_at = time.perf_counter()
        try:
            validated = validate_frame(frame)
        except CVException:
            validated = np.zeros((64, 64, 3), dtype=np.uint8)

        if validated.ndim == 2:
            image = cv2.cvtColor(validated, cv2.COLOR_GRAY2BGR)
        else:
            image = validated

        height, width = image.shape[:2]
        binary = create_lane_binary(image, self.config)
        M, Minv = get_perspective_matrices(width, height, self.config)
        warped_binary = warp_perspective(binary.astype(np.uint8), M, width, height)

        left_base, right_base = compute_histogram_lane_bases(warped_binary)
        left_x_pixels, left_y_pixels, right_x_pixels, right_y_pixels = sliding_window_search(warped_binary, window_count=self.config.sliding_window_count, margin=self.config.window_margin)

        left_lane_poly = fit_polynomial_lane(left_x_pixels, left_y_pixels, min_pixels=self.config.min_pixels_fit) if len(left_x_pixels) >= self.config.min_pixels_fit else None
        right_lane_poly = fit_polynomial_lane(right_x_pixels, right_y_pixels, min_pixels=self.config.min_pixels_fit) if len(right_x_pixels) >= self.config.min_pixels_fit else None

        left_lane = None
        right_lane = None
        if left_lane_poly is not None and validate_polynomial_lane(left_lane_poly, width, height):
            left_lane = left_lane_poly
        if right_lane_poly is not None and validate_polynomial_lane(right_lane_poly, width, height):
            right_lane = right_lane_poly

        # The polynomials live in the bird's-eye (warped) image. Positions, the lane centre and the vehicle offset are
        # reported in the camera image, at the shared reference row, by mapping warped points back through Minv.
        frame_center_x = float(width / 2.0)
        reference_y = reference_row(height)
        warped_reference_y = float(_transform_points(np.array([[frame_center_x, reference_y]]), M)[0, 1])
        left_x = _image_x_at(left_lane, warped_reference_y, Minv)
        right_x = _image_x_at(right_lane, warped_reference_y, Minv)
        lane_width = right_x - left_x if left_x is not None and right_x is not None else None
        if lane_width is not None and lane_width <= 0:
            left_x = right_x = lane_width = None  # boundaries crossed: not a lane
        lane_center_x = calculate_lane_center(left_x, right_x)
        vehicle_offset_pixels = calculate_vehicle_offset(frame_center_x, lane_center_x)
        vehicle_offset_normalized = calculate_vehicle_offset_normalized(vehicle_offset_pixels, lane_width)
        left_curvature = calculate_polynomial_curvature(left_lane.coefficients) if left_lane is not None else float("nan")
        right_curvature = calculate_polynomial_curvature(right_lane.coefficients) if right_lane is not None else float("nan")
        average_curvature = float(np.nanmean([left_curvature, right_curvature])) if np.isfinite(left_curvature) and np.isfinite(right_curvature) else float("nan")

        confidence = 0.0
        if left_lane is not None or right_lane is not None:
            confidence = 0.3
            if left_lane is not None and right_lane is not None:
                confidence += 0.3
            if lane_width is not None:
                confidence += 0.2
            if np.isfinite(average_curvature):
                confidence += 0.2
        confidence = float(min(1.0, max(0.0, confidence)))

        overlay = image.copy()
        polygon = None
        left_line = _lane_line(left_lane, warped_reference_y, Minv, confidence)
        right_line = _lane_line(right_lane, warped_reference_y, Minv, confidence)
        if left_lane is not None and right_lane is not None and lane_width is not None:
            top = max(float(left_lane.plot_y.min()), float(right_lane.plot_y.min()))
            ys = np.linspace(top, warped_reference_y, 20)
            left_points = _transform_points(np.stack([np.polyval(left_lane.coefficients, ys), ys], axis=1), Minv)
            right_points = _transform_points(np.stack([np.polyval(right_lane.coefficients, ys), ys], axis=1), Minv)
            polygon = np.concatenate([left_points, right_points[::-1]]).round().astype(np.int32)
            fill = overlay.copy()
            cv2.fillPoly(fill, [polygon], (0, 255, 0))
            overlay = cv2.addWeighted(overlay, 1.0 - self.config.overlay_opacity, fill, self.config.overlay_opacity, 0)
        for points in (_curve_points(left_lane, warped_reference_y, Minv), _curve_points(right_lane, warped_reference_y, Minv)):
            if points is not None:
                cv2.polylines(overlay, [points], False, (0, 255, 0), 3)

        warnings: list[str] = []
        if left_lane is None and right_lane is None:
            warnings.append("No lane boundary was detected")
        elif left_lane is None or right_lane is None:
            warnings.append("Only one lane boundary was detected")
        elif vehicle_offset_pixels is None:
            warnings.append("Vehicle offset unavailable: the lane boundaries cross above the reference row")

        return LaneDetectionResult(
            success=True,
            processed_frame=overlay,
            left_lane=left_line,
            right_lane=right_line,
            lane_polygon=polygon,
            lane_center_x=lane_center_x,
            frame_center_x=frame_center_x,
            vehicle_offset_pixels=vehicle_offset_pixels,
            vehicle_offset_normalized=vehicle_offset_normalized,
            confidence=float(confidence),
            processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
            detected_segment_count=int(len(left_x_pixels) + len(right_x_pixels)),
            lane_width_bottom_pixels=lane_width,
            left_curvature_pixels=left_curvature if np.isfinite(left_curvature) else None,
            right_curvature_pixels=right_curvature if np.isfinite(right_curvature) else None,
            average_curvature_pixels=average_curvature if np.isfinite(average_curvature) else None,
            warnings=warnings,
            debug_output={"binary": binary, "warped_binary": warped_binary},
        )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def _image_x_at(lane: PolynomialLane | None, warped_y: float, Minv: np.ndarray) -> float | None:
    if lane is None:
        return None
    warped_x = float(np.polyval(lane.coefficients, warped_y))
    return float(_transform_points(np.array([[warped_x, warped_y]]), Minv)[0, 0])


def _curve_points(lane: PolynomialLane | None, warped_reference_y: float, Minv: np.ndarray) -> np.ndarray | None:
    if lane is None:
        return None
    ys = np.linspace(float(lane.plot_y.min()), warped_reference_y, 20)
    return _transform_points(np.stack([np.polyval(lane.coefficients, ys), ys], axis=1), Minv).round().astype(np.int32)


def _lane_line(lane: PolynomialLane | None, warped_reference_y: float, Minv: np.ndarray, confidence: float) -> LaneLine | None:
    """Straight summary of a curved boundary in image coordinates: far end (x1, y1) to the reference row (x2, y2)."""
    if lane is None:
        return None
    top_y = float(lane.plot_y.min())
    (x1, y1), (x2, y2) = _transform_points(np.array([[np.polyval(lane.coefficients, top_y), top_y],
                                                     [np.polyval(lane.coefficients, warped_reference_y), warped_reference_y]]), Minv)
    slope = (y2 - y1) / (x2 - x1) if abs(x2 - x1) > 1e-6 else 0.0
    return LaneLine(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), slope=float(slope), confidence=float(confidence))
