from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from ..core.config import get_settings
from ..core.logging import get_logger
from ..cv.detectors import get_detector
from ..cv.temporal.smoother import TemporalLaneSmoother
from ..schemas.video_detection import (
    VideoDetectionMetrics,
    VideoDetectionResponse,
    VideoDetectionResultReference,
    VideoMediaReference,
)
from ..services.history_service import HistoryService
from ..utils.video_validation import VideoValidationError, validate_video_upload

logger = get_logger("lanevision.video_processing_service")
settings = get_settings()


class VideoProcessingService:
    """Process uploaded videos frame-by-frame with the classical detector and temporal smoothing."""

    def __init__(self) -> None:
        self.detector = get_detector("classic")
        self.history_service = HistoryService(database_url=settings.database_url)
        self.history_service.initialize()

    def _make_output_path(self, extension: str, prefix: str = "video") -> Path:
        suffix = ".mp4" if extension.lower() == ".mp4" else extension.lower()
        return Path(settings.processed_dir) / f"{prefix}-{uuid.uuid4()}{suffix}"

    def _build_writer(self, output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter | None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            fallback_fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(output_path), fallback_fourcc, fps, (width, height))
        return writer if writer.isOpened() else None

    def _validate_processed_frame(self, frame: np.ndarray, expected_width: int, expected_height: int) -> bool:
        if not isinstance(frame, np.ndarray):
            return False
        if frame.ndim != 3:
            return False
        if frame.shape[0] != expected_height or frame.shape[1] != expected_width:
            return False
        if frame.dtype != np.uint8:
            return False
        if frame.shape[2] != 3:
            return False
        return True

    def _validate_output_artifact(self, output_path: Path, expected_width: int, expected_height: int, expected_fps: float | None) -> tuple[bool, str | None, int | None, int | None, tuple[int, int] | None, float | None]:
        if not output_path.exists() or output_path.stat().st_size <= 0:
            return False, "Output artifact was not created", None, None, None, None

        capture = cv2.VideoCapture(str(output_path))
        if not capture.isOpened():
            return False, "Output artifact could not be opened by OpenCV", None, None, None, None

        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                return False, "Output artifact contains no decodable frames", None, None, None, None
            if not self._validate_processed_frame(frame, expected_width, expected_height):
                return False, "Decoded output frame has invalid dimensions or dtype", None, None, None, None

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                return False, "Decoded output frame has invalid dimensions", None, None, None, None
            if expected_fps is not None and fps is not None and fps <= 0:
                return False, "Decoded output has invalid FPS metadata", None, None, None, None
            return True, None, frame_count, int(frame.shape[1]), (width, height), float(fps) if fps is not None else None
        finally:
            capture.release()

    def _cleanup_artifact(self, artifact: Path | None) -> None:
        if artifact is None:
            return
        try:
            artifact.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to remove temporary artifact %s: %s", artifact, exc)

    def process_upload(self, file: UploadFile, algorithm: str = "classic") -> VideoDetectionResponse:
        if file is None:
            raise VideoValidationError(400, "No file provided")

        upload_dir = Path(settings.upload_dir)
        processed_dir = Path(settings.processed_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_handle:
            temp_path = Path(temp_handle.name)
            size_bytes = 0
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                temp_handle.write(chunk)
                if size_bytes > settings.max_video_upload_size_mb * 1024 * 1024:
                    break

        temp_output_path: Path | None = None
        final_output_path: Path | None = None
        writer: cv2.VideoWriter | None = None
        capture: cv2.VideoCapture | None = None
        started_at = time.perf_counter()
        upload_started_at = time.perf_counter()
        validation_started_at = None
        capture_started_at = None
        processing_started_at = None
        writer_started_at = None
        finalization_started_at = None

        try:
            validation_started_at = time.perf_counter()
            validated = validate_video_upload(file, temp_path, size_bytes)
            if size_bytes > settings.max_video_upload_size_mb * 1024 * 1024:
                raise VideoValidationError(413, "File too large")

            safe_name = f"{uuid.uuid4()}{validated['extension']}"
            original_path = upload_dir / safe_name
            shutil.copyfile(str(temp_path), str(original_path))

            capture_started_at = time.perf_counter()
            capture = cv2.VideoCapture(str(original_path))
            if not capture.isOpened():
                raise VideoValidationError(400, "Unable to read uploaded video")

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = frame_count / fps if fps and fps > 0 else 0.0
            if width <= 0 or height <= 0:
                raise VideoValidationError(400, "Invalid video dimensions")
            if fps is None or fps <= 0:
                fps = 24.0

            temp_output_path = Path(settings.processed_dir) / f"{uuid.uuid4()}.processing.mp4"
            writer_started_at = time.perf_counter()
            writer = self._build_writer(temp_output_path, width, height, float(fps))
            if writer is None:
                raise VideoValidationError(500, "Unable to initialize video writer")

            smoother = TemporalLaneSmoother(alpha=float(settings.temporal_smoothing_alpha), max_missing_frames=settings.max_missing_lane_frames)
            frames_read = 0
            frames_processed = 0
            frames_written = 0
            successful_detection_frames = 0
            left_lane_detection_frames = 0
            right_lane_detection_frames = 0
            both_lanes_detection_frames = 0
            offset_values: list[float] = []
            processing_times: list[float] = []
            confidences: list[float] = []
            warnings: list[str] = []

            processing_started_at = time.perf_counter()
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frames_read += 1
                started_at = time.perf_counter()
                detector = get_detector(algorithm)
                detection = detector.detect(frame)
                processing_times.append((time.perf_counter() - started_at) * 1000.0)
                confidences.append(float(detection.confidence))

                left_lane = None
                right_lane = None
                if detection.left_lane is not None:
                    left_lane = (float(detection.left_lane.x1), float(detection.left_lane.y1), float(detection.left_lane.x2), float(detection.left_lane.y2))
                if detection.right_lane is not None:
                    right_lane = (float(detection.right_lane.x1), float(detection.right_lane.y1), float(detection.right_lane.x2), float(detection.right_lane.y2))

                smoothed_state = smoother.update(left_lane, right_lane)
                if smoothed_state.left is not None:
                    left_lane = smoothed_state.left
                if smoothed_state.right is not None:
                    right_lane = smoothed_state.right

                if detection.left_lane is not None or detection.right_lane is not None:
                    successful_detection_frames += 1
                if left_lane is not None:
                    left_lane_detection_frames += 1
                if right_lane is not None:
                    right_lane_detection_frames += 1
                if left_lane is not None and right_lane is not None:
                    both_lanes_detection_frames += 1

                if detection.lane_center_x is not None:
                    offset_values.append(float(detection.lane_center_x))

                overlay_frame = detection.processed_frame.copy()
                if not self._validate_processed_frame(overlay_frame, width, height):
                    raise VideoValidationError(500, "Detected frame was not converted into a valid video frame")

                cv2.putText(overlay_frame, "LaneVision AI", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(overlay_frame, f"Algorithm: {algorithm.title()}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(overlay_frame, f"Confidence: {detection.confidence:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(overlay_frame, f"Offset: {detection.vehicle_offset_pixels or 0:.1f}px", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(overlay_frame, f"FPS: {float(fps):.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                writer.write(overlay_frame)
                frames_processed += 1
                frames_written += 1

            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
            finalization_started_at = time.perf_counter()

            if frames_read <= 0:
                raise VideoValidationError(400, "No readable frames were found in the uploaded video")
            if frames_processed <= 0:
                raise VideoValidationError(500, "No frames were processed into the output video")
            if frames_written <= 0:
                raise VideoValidationError(500, "No frames were written to the output video")

            final_output_path = self._make_output_path(validated["extension"])
            os.replace(temp_output_path, final_output_path)

            valid_output, validation_error, validated_frame_count, validated_width, validated_size, validated_fps = self._validate_output_artifact(final_output_path, width, height, float(fps))
            if not valid_output:
                self._cleanup_artifact(final_output_path)
                raise VideoValidationError(500, validation_error or "Processed video failed validation")

            average_processing_fps = float(frames_processed / max(sum(processing_times) / 1000.0, 1e-6)) if frames_processed else 0.0
            average_processing_time_ms = float(sum(processing_times) / len(processing_times)) if processing_times else 0.0
            average_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
            detection_success_rate = float(successful_detection_frames / frames_processed) if frames_processed else 0.0
            both_lane_detection_rate = float(both_lanes_detection_frames / frames_processed) if frames_processed else 0.0
            average_vehicle_offset_pixels = float(sum(offset_values) / len(offset_values)) if offset_values else None

            total_request_time_ms = (time.perf_counter() - started_at) * 1000.0
            logger.info(
                "Video processing completed | algorithm=%s | frames_read=%d | frames_processed=%d | frames_written=%d | output_size=%s | total_ms=%.2f | validation_ms=%.2f | capture_ms=%.2f | processing_ms=%.2f | writer_ms=%.2f | finalization_ms=%.2f",
                algorithm,
                frames_read,
                frames_processed,
                frames_written,
                final_output_path.stat().st_size if final_output_path.exists() else 0,
                total_request_time_ms,
                (validation_started_at - upload_started_at) * 1000.0 if validation_started_at is not None else 0.0,
                (processing_started_at - capture_started_at) * 1000.0 if processing_started_at is not None and capture_started_at is not None else 0.0,
                (time.perf_counter() - processing_started_at) * 1000.0 if processing_started_at is not None else 0.0,
                (writer_started_at - capture_started_at) * 1000.0 if writer_started_at is not None and capture_started_at is not None else 0.0,
                (finalization_started_at - processing_started_at) * 1000.0 if finalization_started_at is not None and processing_started_at is not None else 0.0,
            )

            response = VideoDetectionResponse(
                id=str(uuid.uuid4()),
                status="success",
                algorithm=algorithm,
                original=VideoMediaReference(
                    filename=file.filename or "video",
                    url=f"/media/uploads/{safe_name}",
                    width=width,
                    height=height,
                    fps=float(fps),
                    frame_count=frame_count,
                    duration_seconds=float(duration_seconds),
                ),
                result=VideoDetectionResultReference(url=f"/media/processed/{final_output_path.name}"),
                metrics=VideoDetectionMetrics(
                    frames_processed=frames_processed,
                    source_fps=float(fps),
                    average_processing_fps=average_processing_fps,
                    average_processing_time_ms=average_processing_time_ms,
                    average_vehicle_offset_pixels=average_vehicle_offset_pixels,
                    average_confidence=average_confidence,
                    detection_success_rate=detection_success_rate,
                    both_lane_detection_rate=both_lane_detection_rate,
                ),
                warnings=warnings,
            )
            self.history_service.record_detection(
                kind="video",
                algorithm=algorithm,
                status="success",
                filename=file.filename or "video",
                original_url=f"/media/uploads/{safe_name}",
                result_url=response.result.url,
                metrics={
                    "confidence": float(response.metrics.average_confidence or 0.0),
                    "processing_time_ms": float(response.metrics.average_processing_time_ms or 0.0),
                    "frames_processed": int(response.metrics.frames_processed),
                    "detection_success_rate": float(response.metrics.detection_success_rate or 0.0),
                },
                warnings=warnings,
            )
            return response
        except VideoValidationError as exc:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
            self._cleanup_artifact(temp_output_path)
            self._cleanup_artifact(final_output_path)
            self.history_service.record_detection(
                kind="video",
                algorithm=algorithm,
                status="failed",
                filename=file.filename or "video",
                original_url=f"/media/uploads/{uuid.uuid4()}{Path(file.filename or 'video').suffix}",
                result_url="/media/processed/failed.mp4",
                metrics={"confidence": 0.0, "processing_time_ms": 0.0},
                warnings=[exc.detail],
            )
            raise
        except Exception as exc:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
            self._cleanup_artifact(temp_output_path)
            self._cleanup_artifact(final_output_path)
            logger.exception("Video processing failed for %s", file.filename or "upload")
            self.history_service.record_detection(
                kind="video",
                algorithm=algorithm,
                status="failed",
                filename=file.filename or "video",
                original_url=f"/media/uploads/{uuid.uuid4()}{Path(file.filename or 'video').suffix}",
                result_url="/media/processed/failed.mp4",
                metrics={"confidence": 0.0, "processing_time_ms": 0.0},
                warnings=[str(exc)],
            )
            raise VideoValidationError(500, str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)
