from __future__ import annotations

import time

import numpy as np

from ..cv.types import LaneDetectionResult, LaneLine
from .config import AIConfig
from .exceptions import AIModelError, AIUnavailableError
from .inference import AIInferenceEngine
from .lane_extraction import LaneExtractor
from .overlay import OverlayRenderer
from .postprocessing import MaskPostprocessor


class AILaneDetectorAdapter:
    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig.from_settings()
        self.engine = AIInferenceEngine(self.config)
        self.postprocessor = MaskPostprocessor(self.config)
        self.extractor = LaneExtractor()
        self.overlay_renderer = OverlayRenderer(self.config)

    def detect(self, frame: np.ndarray, debug: bool = False) -> LaneDetectionResult:
        started_at = time.perf_counter()
        if frame is None:
            raise AIModelError("Input frame is required")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise AIModelError("Expected a 3-channel BGR image")

        try:
            inference_result = self.engine.run(frame)
            # MaskPostprocessor expects (height, width); inference_result["original_size"] is (width, height).
            mask = self.postprocessor.process(inference_result["mask"], frame.shape[:2])
            left_lane, right_lane, _ = self.extractor.extract(mask, frame)
            overlay = self.overlay_renderer.render(frame, mask)
            confidence = self._calculate_confidence(mask, left_lane, right_lane)
            return LaneDetectionResult(
                success=True,
                processed_frame=overlay,
                left_lane=left_lane,
                right_lane=right_lane,
                lane_polygon=None,
                lane_center_x=self._calculate_lane_center(left_lane, right_lane, frame.shape[1]),
                frame_center_x=float(frame.shape[1]) / 2.0,
                vehicle_offset_pixels=self._calculate_vehicle_offset(left_lane, right_lane, frame.shape[1]),
                vehicle_offset_normalized=None,
                confidence=float(confidence),
                processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
                detected_segment_count=int(mask.sum() // 10),
                warnings=[],
                debug_output={"mask": mask, "model_metadata": inference_result["model_metadata"]},
            )
        except AIUnavailableError as exc:
            warning = f"AI model unavailable: {exc}"
            status_payload = self.engine.loader.get_status()
            return LaneDetectionResult(
                success=True,
                processed_frame=frame.copy(),
                left_lane=None,
                right_lane=None,
                lane_polygon=None,
                lane_center_x=None,
                frame_center_x=float(frame.shape[1]) / 2.0,
                vehicle_offset_pixels=None,
                vehicle_offset_normalized=None,
                confidence=0.0,
                processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
                detected_segment_count=0,
                warnings=[warning],
                debug_output={
                    "model_status": status_payload,
                    "model_metadata": {
                        "model_name": status_payload.get("model_name"),
                        "architecture": status_payload.get("architecture"),
                        "device": status_payload.get("device"),
                        "checkpoint_loaded": status_payload.get("checkpoint_loaded", False),
                        "input_size": status_payload.get("input_size"),
                    },
                },
            )
        except Exception as exc:
            raise AIModelError(f"AI inference failed: {exc}") from exc

    def _calculate_confidence(self, mask: np.ndarray, left_lane: LaneLine | None, right_lane: LaneLine | None) -> float:
        if mask is None or mask.size == 0:
            return 0.0
        lane_pixels = float(mask.sum())
        base = min(1.0, lane_pixels / max(mask.size, 1))
        geometry_bonus = 0.0
        if left_lane is not None:
            geometry_bonus += 0.1
        if right_lane is not None:
            geometry_bonus += 0.1
        return min(0.99, base + geometry_bonus)

    def _calculate_lane_center(self, left_lane: LaneLine | None, right_lane: LaneLine | None, width: int) -> float | None:
        if left_lane is None and right_lane is None:
            return None
        if left_lane is not None and right_lane is not None:
            return float((left_lane.x2 + right_lane.x2) / 2.0)
        lane = left_lane or right_lane
        return float(lane.x2)

    def _calculate_vehicle_offset(self, left_lane: LaneLine | None, right_lane: LaneLine | None, width: int) -> float | None:
        lane_center_x = self._calculate_lane_center(left_lane, right_lane, width)
        if lane_center_x is None:
            return None
        frame_center_x = width / 2.0
        return float(lane_center_x - frame_center_x)
