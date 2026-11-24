from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..cv.types import LaneDetectionResult, LaneLine
from .config import AIConfig
from .exceptions import AIModelError, AIUnavailableError
from .inference import AIInferenceEngine
from .lane_geometry import EgoLane, LaneCurve, LaneGeometryConfig, extract_lanes, select_ego_lane
from .lane_tracking import LaneTracker
from .overlay import OverlayRenderer
from .postprocessing import MaskPostprocessor


def _lane_line(curve: LaneCurve, bottom_y: float, confidence: float) -> LaneLine:
    """Straight summary of a lane from its top observed point to the reference row (x2, y2 is the near end)."""
    x1, y1 = curve.x_at(curve.y_top), curve.y_top
    x2, y2 = curve.x_at(bottom_y), bottom_y
    slope = (y2 - y1) / (x2 - x1) if abs(x2 - x1) > 1e-6 else 0.0
    return LaneLine(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), slope=float(slope), confidence=float(confidence))


def _ego_polygon(ego: EgoLane, step: float = 10.0) -> np.ndarray | None:
    if not ego.reliable or ego.left is None or ego.right is None:
        return None
    top = max(ego.left.y_top, ego.right.y_top)
    ys = np.arange(top, ego.reference_y + 1e-6, step)
    if len(ys) < 2:
        return None
    left = [(ego.left.x_at(y), y) for y in ys]
    right = [(ego.right.x_at(y), y) for y in ys[::-1]]
    return np.array(left + right, dtype=np.float32).round().astype(np.int32)


class AILaneDetectorAdapter:
    """Run the segmentation model and turn its output into lanes, the ego lane and the vehicle offset."""

    def __init__(self, config: AIConfig | None = None, geometry: LaneGeometryConfig | None = None) -> None:
        self.config = config or AIConfig.from_settings()
        self.geometry = geometry or LaneGeometryConfig()
        self.engine = AIInferenceEngine(self.config)
        self.postprocessor = MaskPostprocessor(self.config)
        self.overlay_renderer = OverlayRenderer(self.config)

    def model_status(self) -> dict[str, Any]:
        """What actually backs this detector: device, whether the checkpoint is loaded, and any load error."""
        status = self.engine.loader.get_status()
        info: dict[str, Any] = {
            "model_name": status.get("model_name"),
            "architecture": status.get("architecture"),
            "device": status.get("device"),
            "checkpoint_loaded": bool(status.get("checkpoint_loaded")),
            "checkpoint": Path(self.config.model_path).name if self.config.model_path else None,
            "input_size": status.get("input_size"),
        }
        if status.get("error"):
            info["error"] = status["error"]
        return info

    def require_model(self) -> None:
        """Load the model now (once per process) or raise AIUnavailableError / AIModelError saying why it cannot be.

        The API calls this before processing so a request for AI inference without a usable checkpoint fails with an
        explicit error instead of returning unprocessed frames."""
        self.engine.loader.load()

    def unavailable_detail(self, exc: Exception) -> str:
        """User-facing explanation of why the model cannot serve requests (file name only, never the full path)."""
        detail = f"AI model unavailable: {exc}"
        model_path = self.config.model_path
        if self.config.enabled and (not model_path or not Path(model_path).is_file()):
            if model_path:
                detail += f" ({Path(model_path).name})"
            detail += ". Set AI_MODEL_PATH to a trained checkpoint file (see backend/app/ai/README.md)."
        return detail

    def detect(self, frame: np.ndarray, debug: bool = False, tracker: LaneTracker | None = None) -> LaneDetectionResult:
        """Detect lanes in one frame. In a video, pass the video's LaneTracker: lanes then keep their identity across
        frames, are smoothed, survive short gaps, and the ego lane is chosen from the tracked lanes. The adapter is
        shared between requests and keeps no per-video state itself."""
        started_at = time.perf_counter()
        if frame is None:
            raise AIModelError("Input frame is required")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise AIModelError("Expected a 3-channel BGR image")
        height, width = frame.shape[:2]

        try:
            inference_result = self.engine.run(frame)
            probability = cv2.resize(inference_result["probability_mask"], (width, height), interpolation=cv2.INTER_LINEAR)
            mask = self.postprocessor.process(inference_result["mask"], (height, width))
            detected = extract_lanes(probability, self.geometry)
            tracking = None
            if tracker is None:
                lanes = detected
                held: set[int] = set()
                ego = select_ego_lane(lanes, width, height, self.geometry)
            else:
                tracked, events = tracker.update(detected, width, height)
                lanes = [lane.curve for lane in tracked]
                held = {id(lane.curve) for lane in tracked if lane.held}
                ego = select_ego_lane(lanes, width, height, self.geometry)
                ego_ids = tracker.note_ego(tracked, ego, events)
                tracking = {
                    "lanes": [{"id": lane.track_id, "held_frames": lane.missed, "detections": lane.hits} for lane in tracked],
                    "detected_lanes": len(detected),
                    "ego_lane_ids": ego_ids,
                    "appeared": events.appeared,
                    "disappeared": events.disappeared,
                    "ego_lane_change": events.ego_lane_change,
                    "ego_boundary_held": any(id(boundary) in held for boundary in (ego.left, ego.right) if boundary is not None),
                }
            polygon = _ego_polygon(ego)
            overlay = self.overlay_renderer.render(frame, mask, lanes=lanes, ego=ego, polygon=polygon, held=held)
            warnings = [] if ego.reliable else [f"Vehicle offset unavailable: {ego.reason}"]
            if tracking is not None and ego.reliable and tracking["ego_boundary_held"]:
                warnings.append("Ego-lane boundary held from previous frames (not detected in this frame)")
            left_confidence = ego.left.confidence(height, self.geometry.top_ratio) if ego.left else 0.0
            right_confidence = ego.right.confidence(height, self.geometry.top_ratio) if ego.right else 0.0
            return LaneDetectionResult(
                success=True,
                processed_frame=overlay,
                left_lane=_lane_line(ego.left, ego.reference_y, left_confidence) if ego.left else None,
                right_lane=_lane_line(ego.right, ego.reference_y, right_confidence) if ego.right else None,
                lane_polygon=polygon,
                lane_center_x=ego.lane_center_x,
                frame_center_x=ego.camera_x,
                vehicle_offset_pixels=ego.offset_pixels,
                vehicle_offset_normalized=ego.offset_normalized,
                confidence=float(ego.confidence),
                processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
                detected_segment_count=len(lanes),
                lane_width_bottom_pixels=ego.lane_width if ego.reliable else None,
                warnings=warnings,
                debug_output={"mask": mask, "probability": probability, "lanes": lanes, "ego_lane": ego, "tracking": tracking,
                              "model_metadata": inference_result["model_metadata"]},
            )
        except AIUnavailableError as exc:
            status_payload = self.model_status()
            return LaneDetectionResult(
                success=True,
                processed_frame=frame.copy(),
                left_lane=None,
                right_lane=None,
                lane_polygon=None,
                lane_center_x=None,
                frame_center_x=float(width) / 2.0,
                vehicle_offset_pixels=None,
                vehicle_offset_normalized=None,
                confidence=0.0,
                processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
                detected_segment_count=0,
                warnings=[f"AI model unavailable: {exc}"],
                debug_output={"model_status": status_payload, "model_metadata": status_payload},
            )
        except Exception as exc:
            raise AIModelError(f"AI inference failed: {exc}") from exc
