from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from .config import LaneDetectionConfig
from .metrics.lane_metrics import (
    calculate_confidence,
    calculate_lane_center,
    calculate_vehicle_offset,
    calculate_vehicle_offset_normalized,
    line_x_at,
    reference_row,
)
from .stages.color_filter import apply_lane_color_filter
from .stages.edges import detect_edges
from .stages.hough import detect_hough_lines
from .stages.lane_fitting import fit_lane_line, is_horizontal_line
from .stages.overlay import draw_lane_overlay
from .stages.preprocessing import gaussian_blur, resize_if_required, to_grayscale
from .stages.roi import apply_roi
from .types import LaneDetectionResult, LaneLine, LaneSegment
from .utils.validation import CVException, validate_frame


def _scale_lane(lane: LaneLine | None, scale_x: float, scale_y: float) -> LaneLine | None:
    if lane is None or (scale_x == 1.0 and scale_y == 1.0):
        return lane
    return LaneLine(x1=lane.x1 * scale_x, y1=lane.y1 * scale_y, x2=lane.x2 * scale_x, y2=lane.y2 * scale_y,
                    slope=lane.slope * scale_y / scale_x, confidence=lane.confidence, support_segments=lane.support_segments)


class ClassicLaneDetector:
    """Orchestrate the classical computer-vision lane detection pipeline."""

    def __init__(self, config: LaneDetectionConfig | None = None) -> None:
        self.config = config or LaneDetectionConfig()

    def detect(self, frame: np.ndarray, debug: bool = False) -> LaneDetectionResult:
        """Run the lane detection pipeline on an input frame."""
        started_at = time.perf_counter()
        validated = validate_frame(frame)
        resized = resize_if_required(validated, self.config)
        grayscale = to_grayscale(resized)
        blurred = gaussian_blur(grayscale, self.config)
        color_mask = apply_lane_color_filter(resized, self.config)
        edges = detect_edges(resized, self.config)
        roi_edges = apply_roi(edges, self.config)
        raw_segments = detect_hough_lines(roi_edges, self.config)

        filtered_segments: list[LaneSegment] = []
        for segment in raw_segments:
            if is_horizontal_line(segment, self.config):
                continue
            if abs(segment.slope) < self.config.min_slope_abs:
                continue
            if segment.length is None or segment.length < self.config.min_segment_length:
                continue
            filtered_segments.append(segment)

        # A segment supports the boundary on its side of the frame only if it leans the way that boundary does in the
        # image (y grows downwards): the left boundary rises to the right (negative slope), the right one to the left.
        half_width = resized.shape[1] / 2.0
        left_segments = [segment for segment in filtered_segments if (segment.x1 + segment.x2) / 2.0 < half_width and segment.slope < 0]
        right_segments = [segment for segment in filtered_segments if (segment.x1 + segment.x2) / 2.0 > half_width and segment.slope > 0]
        roi_ys = [y for _, y in self.config.roi_polygon]
        roi_top_y = int(round(min(roi_ys) * resized.shape[0]))
        roi_bottom_y = int(round(max(roi_ys) * resized.shape[0]))
        left_lane = fit_lane_line(left_segments, resized.shape[1], resized.shape[0], roi_top_y, roi_bottom_y)
        right_lane = fit_lane_line(right_segments, resized.shape[1], resized.shape[0], roi_top_y, roi_bottom_y)

        # Report geometry in the coordinates of the input frame (the pipeline may have downscaled it).
        scale_x = validated.shape[1] / resized.shape[1]
        scale_y = validated.shape[0] / resized.shape[0]
        left_lane = _scale_lane(left_lane, scale_x, scale_y)
        right_lane = _scale_lane(right_lane, scale_x, scale_y)
        frame_height, frame_width = validated.shape[:2]
        overlay, polygon = draw_lane_overlay(validated, left_lane, right_lane, self.config)

        frame_center_x = frame_width / 2.0
        reference_y = reference_row(frame_height)
        left_x = line_x_at(left_lane.x1, left_lane.y1, left_lane.x2, left_lane.y2, reference_y) if left_lane is not None else None
        right_x = line_x_at(right_lane.x1, right_lane.y1, right_lane.x2, right_lane.y2, reference_y) if right_lane is not None else None
        lane_center_x = calculate_lane_center(left_x, right_x)
        lane_width = right_x - left_x if left_x is not None and right_x is not None else None
        if lane_width is not None and lane_width <= 0:
            lane_center_x, lane_width = None, None  # the fitted boundaries cross before the reference row
        vehicle_offset_pixels = calculate_vehicle_offset(frame_center_x, lane_center_x)
        vehicle_offset_normalized = calculate_vehicle_offset_normalized(vehicle_offset_pixels, lane_width)

        confidence = calculate_confidence(
            supporting_segment_count=float(len(filtered_segments)),
            supporting_segment_length=float(sum(segment.length or 0.0 for segment in filtered_segments)),
            left_lane_present=left_lane is not None,
            right_lane_present=right_lane is not None,
            slope_consistency=0.8 if left_lane is not None and right_lane is not None else 0.5,
        )

        if left_lane is None and right_lane is None:
            confidence = 0.0

        warnings: list[str] = []
        if left_lane is None and right_lane is None:
            warnings.append("No lane boundary was detected")
        elif left_lane is None or right_lane is None:
            warnings.append("Only one lane boundary was detected")
        elif vehicle_offset_pixels is None:
            warnings.append("Vehicle offset unavailable: the lane boundaries cross above the reference row")
        if not filtered_segments:
            warnings.append("No Hough line segments met the filtering criteria")

        debug_output: dict[str, np.ndarray] | None = None
        if debug:
            debug_output = {
                "color_mask": color_mask,
                "blurred": blurred,
                "edges": edges,
                "roi_edges": roi_edges,
                "raw_hough_segments": cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY),
                "filtered_segments": cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY),
                "final_overlay": overlay,
            }

        return LaneDetectionResult(
            success=True,
            processed_frame=overlay,
            left_lane=left_lane,
            right_lane=right_lane,
            lane_polygon=polygon,
            lane_center_x=lane_center_x,
            frame_center_x=frame_center_x,
            vehicle_offset_pixels=vehicle_offset_pixels,
            vehicle_offset_normalized=vehicle_offset_normalized,
            confidence=confidence,
            lane_width_bottom_pixels=lane_width,
            processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
            detected_segment_count=len(filtered_segments),
            warnings=warnings,
            debug_output=debug_output,
        )
