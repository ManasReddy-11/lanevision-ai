"""Model caching under concurrency, explicit errors for an unusable AI model, invalid uploads and video timing logs."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.ai.config import AIConfig
from app.ai.model_loader import ModelLoader
from app.ai.models import LaneSegmentationUNet
from app.ai.registry import clear_ai_detectors, get_ai_detector
from app.core.config import get_settings
from app.cv.types import LaneDetectionResult
from app.main import app
from app.services.video_processing_service import VideoProcessingService
from app.utils.video_validation import VideoValidationError

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_registry():
    clear_ai_detectors()
    yield
    clear_ai_detectors()


def _checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "unet.pth"
    torch.save({"state_dict": LaneSegmentationUNet(in_channels=3, num_classes=2).state_dict()}, path)
    return path


def _config(model_path) -> AIConfig:
    return AIConfig(enabled=True, model_path=str(model_path), device="cpu", input_width=64, input_height=32)


def _media_files() -> set[str]:
    settings = get_settings()
    return {str(path) for folder in (settings.upload_dir, settings.processed_dir) if Path(folder).exists() for path in Path(folder).iterdir()}


def _run_concurrently(target, count: int = 8) -> list:
    barrier = threading.Barrier(count)
    results: list = [None] * count
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        barrier.wait()
        try:
            results[index] = target()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors, errors
    return results


def test_model_loader_reads_the_checkpoint_once_under_concurrent_requests(tmp_path, monkeypatch) -> None:
    real_load = torch.load

    def slow_load(*args, **kwargs):  # widen the window in which unsynchronised loads would overlap
        time.sleep(0.05)
        return real_load(*args, **kwargs)

    monkeypatch.setattr("app.ai.model_loader.torch.load", slow_load)
    loader = ModelLoader(_config(_checkpoint(tmp_path)))
    results = _run_concurrently(loader.load)
    assert loader.load_count == 1
    assert all(result is results[0] for result in results)


def test_registry_hands_every_concurrent_request_the_same_detector(tmp_path) -> None:
    config = _config(_checkpoint(tmp_path))
    detectors = _run_concurrently(lambda: get_ai_detector(config))
    assert all(detector is detectors[0] for detector in detectors)


def test_a_failed_load_is_reported_to_every_concurrent_caller_without_rereading(tmp_path) -> None:
    broken = tmp_path / "broken.pth"
    broken.write_bytes(b"not a checkpoint")
    loader = ModelLoader(_config(broken))

    def attempt():
        try:
            loader.load()
        except Exception as exc:
            return exc
        return None

    failures = _run_concurrently(attempt)
    assert all(failure is not None and "Unable to load AI checkpoint" in str(failure) for failure in failures)
    assert loader.load_count == 1


def _png() -> bytes:
    success, encoded = cv2.imencode(".png", np.full((72, 128, 3), 90, dtype=np.uint8))
    assert success
    return encoded.tobytes()


def test_image_api_refuses_ai_inference_without_a_checkpoint(tmp_path, monkeypatch) -> None:
    detector = get_ai_detector(_config(tmp_path / "missing.pth"))
    monkeypatch.setattr("app.services.detection_service.get_detector", lambda mode: detector)
    before = _media_files()

    response = client.post("/api/detect/image", files={"file": ("road.png", _png(), "image/png")}, data={"algorithm": "ai"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail.startswith("AI model unavailable: AI model weights were not found (missing.pth)")
    assert "AI_MODEL_PATH" in detail
    assert str(tmp_path) not in detail  # the file name only, never a machine path
    assert _media_files() == before  # nothing stored for a request that was refused


def test_image_api_reports_a_broken_checkpoint_explicitly(tmp_path, monkeypatch) -> None:
    broken = tmp_path / "broken.pth"
    broken.write_bytes(b"not a checkpoint")
    detector = get_ai_detector(_config(broken))
    monkeypatch.setattr("app.services.detection_service.get_detector", lambda mode: detector)

    response = client.post("/api/detect/image", files={"file": ("road.png", _png(), "image/png")}, data={"algorithm": "ai"})

    assert response.status_code == 503
    assert response.json()["detail"].startswith("AI model unavailable: Unable to load AI checkpoint")


def _video_bytes(tmp_path: Path, frames: int = 3) -> bytes:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for index in range(frames):
        writer.write(np.full((120, 160, 3), 40 * index, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def test_corrupt_video_is_rejected_and_leaves_no_files(tmp_path) -> None:
    before = _media_files()
    response = client.post("/api/detect/video", files={"file": ("clip.mp4", b"\x00\x00\x00 not really an mp4", "video/mp4")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Unable to read uploaded video"
    assert _media_files() == before


def test_video_api_refuses_ai_inference_without_a_checkpoint(tmp_path, monkeypatch) -> None:
    detector = get_ai_detector(_config(tmp_path / "missing.pth"))
    monkeypatch.setattr("app.services.video_processing_service.get_detector", lambda mode: detector)
    before = _media_files()

    response = client.post("/api/detect/video", files={"file": ("clip.mp4", _video_bytes(tmp_path), "video/mp4")}, data={"algorithm": "ai"})

    assert response.status_code == 503
    assert response.json()["detail"].startswith("AI model unavailable: AI model weights were not found")
    assert detector.engine.loader.load_count == 0
    assert _media_files() == before


class _Upload:
    def __init__(self, path: Path) -> None:
        self.filename = path.name
        self.content_type = "video/mp4"
        self.file = open(path, "rb")


def test_video_log_reports_the_whole_request_time(tmp_path, monkeypatch) -> None:
    class SlowDetector:
        def detect(self, frame, debug=False):
            time.sleep(0.03)
            return LaneDetectionResult(success=True, processed_frame=frame.copy(), left_lane=None, right_lane=None, lane_polygon=None,
                                       lane_center_x=None, frame_center_x=80.0, vehicle_offset_pixels=None, vehicle_offset_normalized=None,
                                       confidence=0.0, processing_time_ms=30.0, detected_segment_count=0)

    logged: list[tuple] = []

    class CapturingLogger:
        def info(self, message, *args):
            logged.append((message, args))

        warning = exception = info

    monkeypatch.setattr("app.services.video_processing_service.get_detector", lambda mode: SlowDetector())
    monkeypatch.setattr("app.services.video_processing_service.logger", CapturingLogger())
    path = tmp_path / "clip.mp4"
    path.write_bytes(_video_bytes(tmp_path, frames=5))

    VideoProcessingService().process_upload(_Upload(path), algorithm="classic")

    (message, args), = [entry for entry in logged if entry[0].startswith("Video processing completed")]
    fields = dict(zip([part.split("=")[0].strip() for part in message.split("|")[1:]], args))
    # total_ms used to be measured from the start of the last frame (the per-frame timer reused started_at)
    assert fields["total_ms"] >= fields["processing_ms"] >= 5 * 30.0
