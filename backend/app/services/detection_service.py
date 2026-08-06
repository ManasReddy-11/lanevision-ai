from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from ..core.config import get_settings
from ..core.logging import get_logger
from ..cv.detectors import get_detector
from ..cv.utils.validation import CVException, validate_frame
from ..schemas.detection import DetectionLanes, DetectionMetrics, DetectionResponse, DetectionResultReference, MediaReference
from ..services.history_service import HistoryService
from ..utils.file_validation import UploadValidationError, validate_upload

logger = get_logger("lanevision.detection_service")
settings = get_settings()


class DetectionService:
    """Coordinate image validation, processing, file persistence, and response building."""

    def __init__(self) -> None:
        self.detector = get_detector("classic")
        self.history_service = HistoryService(database_url=settings.database_url)
        self.history_service.initialize()

    async def process_upload(self, file: UploadFile, algorithm: str = "classic") -> DetectionResponse:
        """Validate, decode, process, save, and respond for an uploaded image."""
        if file is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided")

        await file.seek(0)
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty upload")

        file_name = file.filename or "upload"
        try:
            validated = validate_upload(
                file_name=file_name,
                content_type=file.content_type,
                contents=contents,
                max_size_mb=settings.max_upload_size_mb,
            )
        except UploadValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        upload_dir = Path(settings.upload_dir)
        processed_dir = Path(settings.processed_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        safe_name = f"{uuid.uuid4()}{validated['extension']}"
        original_path = upload_dir / safe_name
        original_path.write_bytes(contents)

        frame = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or corrupted image")

        try:
            validated_frame = validate_frame(frame)
        except CVException as exc:
            self.history_service.record_detection(
                kind="image",
                algorithm=algorithm,
                status="failed",
                filename=file_name,
                original_url=f"/media/uploads/{safe_name}",
                result_url=f"/media/processed/{safe_name}",
                metrics={"confidence": 0.0, "processing_time_ms": 0.0},
                warnings=[str(exc)],
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        detector = get_detector(algorithm)
        result = detector.detect(validated_frame)
        processed_path = processed_dir / safe_name
        success = cv2.imwrite(str(processed_path), result.processed_frame)
        if not success:
            logger.error("Failed to write processed image for %s", safe_name)
            self.history_service.record_detection(
                kind="image",
                algorithm=algorithm,
                status="failed",
                filename=file_name,
                original_url=f"/media/uploads/{safe_name}",
                result_url=f"/media/processed/{safe_name}",
                metrics={"confidence": float(result.confidence), "processing_time_ms": float(result.processing_time_ms)},
                warnings=["Failed to save processed image"],
            )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save processed image")

        original_height, original_width = frame.shape[:2]
        ai_metadata = None
        if algorithm == "ai":
            debug_output = getattr(result, "debug_output", None) or {}
            model_metadata = debug_output.get("model_metadata")
            if model_metadata is not None:
                if isinstance(model_metadata, dict):
                    ai_metadata = {
                        "model_name": model_metadata.get("model_name"),
                        "architecture": model_metadata.get("architecture"),
                        "device": model_metadata.get("device"),
                        "checkpoint_loaded": model_metadata.get("checkpoint_loaded"),
                        "input_size": model_metadata.get("input_size"),
                        "inference_time_ms": float(result.processing_time_ms),
                        "confidence": float(result.confidence),
                    }
                else:
                    ai_metadata = {
                        "model_name": getattr(model_metadata, "model_name", None),
                        "architecture": getattr(model_metadata, "architecture", None),
                        "device": getattr(model_metadata, "device", None),
                        "checkpoint_loaded": getattr(model_metadata, "checkpoint_loaded", None),
                        "input_size": [getattr(model_metadata, "input_width", None), getattr(model_metadata, "input_height", None)],
                        "inference_time_ms": float(result.processing_time_ms),
                        "confidence": float(result.confidence),
                    }

        response = DetectionResponse(
            id=str(uuid.uuid4()),
            status="success",
            algorithm=algorithm,
            original=MediaReference(
                filename=file_name,
                url=f"/media/uploads/{safe_name}",
                width=original_width,
                height=original_height,
            ),
            result=DetectionResultReference(url=f"/media/processed/{safe_name}"),
            metrics=DetectionMetrics(
                confidence=float(result.confidence),
                processing_time_ms=float(result.processing_time_ms),
                detected_segments=int(result.detected_segment_count),
                lane_center_x=float(result.lane_center_x) if result.lane_center_x is not None else None,
                frame_center_x=float(result.frame_center_x) if result.frame_center_x is not None else None,
                vehicle_offset_pixels=float(result.vehicle_offset_pixels) if result.vehicle_offset_pixels is not None else None,
                vehicle_offset_normalized=float(result.vehicle_offset_normalized) if result.vehicle_offset_normalized is not None else None,
                lane_width_bottom_pixels=float(result.lane_width_bottom_pixels) if result.lane_width_bottom_pixels is not None else None,
                lane_width_average_pixels=float(result.lane_width_average_pixels) if result.lane_width_average_pixels is not None else None,
                left_curvature_pixels=float(result.left_curvature_pixels) if result.left_curvature_pixels is not None else None,
                right_curvature_pixels=float(result.right_curvature_pixels) if result.right_curvature_pixels is not None else None,
                average_curvature_pixels=float(result.average_curvature_pixels) if result.average_curvature_pixels is not None else None,
            ),
            lanes=DetectionLanes(
                left_detected=result.left_lane is not None,
                right_detected=result.right_lane is not None,
            ),
            warnings=result.warnings,
            ai_metadata=ai_metadata,
        )
        self.history_service.record_detection(
            kind="image",
            algorithm=algorithm,
            status="success",
            filename=file_name,
            original_url=f"/media/uploads/{safe_name}",
            result_url=f"/media/processed/{safe_name}",
            metrics={
                "confidence": float(response.metrics.confidence),
                "processing_time_ms": float(response.metrics.processing_time_ms),
                "detected_segments": int(response.metrics.detected_segments),
                "vehicle_offset_pixels": float(response.metrics.vehicle_offset_pixels) if response.metrics.vehicle_offset_pixels is not None else None,
                "lane_center_x": float(response.metrics.lane_center_x) if response.metrics.lane_center_x is not None else None,
            },
            warnings=response.warnings,
        )
        return response
