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
)
from .stages.color_filter import apply_lane_color_filter
from .stages.edges import detect_edges
from .stages.hough import detect_hough_lines
from .stages.lane_fitting import classify_lane_segments, fit_lane_line, is_horizontal_line
from .stages.overlay import draw_lane_overlay
from .stages.preprocessing import gaussian_blur, resize_if_required, to_grayscale
from .stages.roi import apply_roi
from .types import LaneDetectionResult, LaneLine, LaneSegment
from .utils.validation import CVException, validate_frame


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

        left_segments = [segment for segment in filtered_segments if (segment.x1 + segment.x2) / 2.0 < resized.shape[1] / 2.0]
        right_segments = [segment for segment in filtered_segments if (segment.x1 + segment.x2) / 2.0 > resized.shape[1] / 2.0]
        left_lane, right_lane = classify_lane_segments(left_segments, right_segments, resized.shape[1], resized.shape[0])

        if left_lane is None and right_segments:
            left_lane = fit_lane_line(right_segments, resized.shape[1], resized.shape[0], int(resized.shape[0] * 0.2), resized.shape[0])
        if right_lane is None and left_segments:
            right_lane = fit_lane_line(left_segments, resized.shape[1], resized.shape[0], int(resized.shape[0] * 0.2), resized.shape[0])

        if left_lane is None:
            left_lane = fit_lane_line(left_segments, resized.shape[1], resized.shape[0], int(resized.shape[0] * 0.2), resized.shape[0])
        if right_lane is None:
            right_lane = fit_lane_line(right_segments, resized.shape[1], resized.shape[0], int(resized.shape[0] * 0.2), resized.shape[0])

        overlay, polygon = draw_lane_overlay(resized, left_lane, right_lane, self.config)
        frame_center_x = resized.shape[1] / 2.0
        lane_center_x = calculate_lane_center(
            left_lane.x1 if left_lane is not None else None,
            right_lane.x1 if right_lane is not None else None,
        )
        vehicle_offset_pixels = calculate_vehicle_offset(frame_center_x, lane_center_x)
        vehicle_offset_normalized = calculate_vehicle_offset_normalized(frame_center_x, lane_center_x)

        confidence = calculate_confidence(
            supporting_segment_count=float(len(filtered_segments)),
            supporting_segment_length=float(sum(segment.length or 0.0 for segment in filtered_segments)),
            left_lane_present=left_lane is not None,
            right_lane_present=right_lane is not None,
            slope_consistency=0.8 if left_lane is not None and right_lane is not None else 0.5,
        )

        warnings: list[str] = []
        if left_lane is None or right_lane is None:
            warnings.append("Only one lane boundary was detected")
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
            processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
            detected_segment_count=len(filtered_segments),
            warnings=warnings,
            debug_output=debug_output,
        )
