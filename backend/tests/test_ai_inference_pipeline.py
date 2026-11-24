from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.ai.adapter import AILaneDetectorAdapter
from app.ai.config import AIConfig
from app.ai.models import LaneSegmentationUNet
from app.ai.registry import clear_ai_detectors, get_ai_detector
from app.cv.detectors import get_detector
from app.cv.types import LaneDetectionResult
from app.main import app
from app.services.video_processing_service import VideoProcessingService


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


class _Upload:
    def __init__(self, path: Path) -> None:
        self.filename = path.name
        self.content_type = "video/mp4"
        self.file = open(path, "rb")


def _video(tmp_path: Path, frames: int = 4) -> _Upload:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for index in range(frames):
        writer.write(np.full((120, 160, 3), 40 * index, dtype=np.uint8))
    writer.release()
    return _Upload(path)


def test_ai_detector_is_shared_and_reads_the_checkpoint_once(tmp_path) -> None:
    config = _config(_checkpoint(tmp_path))
    detector = get_ai_detector(config)
    assert get_ai_detector(config) is detector
    for value in (0, 80, 160):
        detector.detect(np.full((72, 128, 3), value, dtype=np.uint8))
    assert detector.engine.loader.load_count == 1


def test_get_detector_returns_the_same_ai_detector_for_every_request() -> None:
    assert get_detector("ai") is get_detector("ai")
    assert get_detector("classic") is not get_detector("classic")


def test_video_loads_the_model_once_and_reports_the_model_that_ran(tmp_path, monkeypatch) -> None:
    detector = get_ai_detector(_config(_checkpoint(tmp_path)))
    monkeypatch.setattr("app.services.video_processing_service.get_detector", lambda mode: detector)

    response = VideoProcessingService().process_upload(_video(tmp_path, frames=5), algorithm="ai")

    assert response.metrics.frames_processed == 5
    assert detector.engine.loader.load_count == 1
    assert response.ai_metadata["checkpoint_loaded"] is True
    assert response.ai_metadata["device"] == "cpu"
    assert response.ai_metadata["checkpoint"] == "unet.pth"
    assert response.ai_metadata["frames_processed"] == 5
    assert 0.0 <= response.metrics.vehicle_offset_available_rate <= 1.0


def test_video_reports_an_unavailable_model_instead_of_claiming_ai_inference(tmp_path, monkeypatch) -> None:
    detector = get_ai_detector(_config(tmp_path / "missing.pth"))
    monkeypatch.setattr("app.services.video_processing_service.get_detector", lambda mode: detector)

    response = VideoProcessingService().process_upload(_video(tmp_path, frames=3), algorithm="ai")

    assert response.ai_metadata["checkpoint_loaded"] is False
    assert "not found" in response.ai_metadata["error"]
    assert response.warnings == ["AI model unavailable: AI model weights were not found (3/3 frames)"]
    assert detector.engine.loader.load_count == 0  # the missing file is detected once, not re-read per frame


def test_video_averages_vehicle_offsets_not_lane_centre_positions(tmp_path, monkeypatch) -> None:
    offsets = iter([12.0, None, -4.0, None])

    class FixedOffsetDetector:
        def detect(self, frame, debug=False):
            offset = next(offsets)
            return LaneDetectionResult(success=True, processed_frame=frame.copy(), left_lane=None, right_lane=None, lane_polygon=None,
                                       lane_center_x=918.0, frame_center_x=80.0, vehicle_offset_pixels=offset, vehicle_offset_normalized=None,
                                       confidence=0.5, processing_time_ms=1.0, detected_segment_count=0,
                                       warnings=[] if offset is not None else ["Vehicle offset unavailable: ego-lane right boundary not detected"])

    monkeypatch.setattr("app.services.video_processing_service.get_detector", lambda mode: FixedOffsetDetector())
    response = VideoProcessingService().process_upload(_video(tmp_path, frames=4), algorithm="classic")

    assert response.metrics.average_vehicle_offset_pixels == pytest.approx(4.0)
    assert response.metrics.vehicle_offset_available_rate == pytest.approx(0.5)
    assert response.warnings == ["Vehicle offset unavailable: ego-lane right boundary not detected (2/4 frames)"]


def test_adapter_measures_the_offset_from_the_ego_lane_of_the_probability_map(tmp_path) -> None:
    adapter = AILaneDetectorAdapter(_config(_checkpoint(tmp_path)))
    probability = np.zeros((160, 320), dtype=np.float32)  # model resolution; lanes meet at (160, 50)
    for bottom_x in (40, 120, 230, 300):
        cv2.line(probability, (160, 50), (bottom_x, 159), 0.9, 2)

    class FixedEngine:
        loader = adapter.engine.loader

        def run(self, frame):
            return {"probability_mask": probability, "mask": (probability >= 0.5).astype(np.uint8), "model_metadata": None}

    adapter.engine = FixedEngine()
    result = adapter.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    def full_res_x(bottom_x: float, y_full: float) -> float:
        y_model = (y_full + 0.5) / 4.5 - 0.5  # cv2.resize maps pixel centres: 160 -> 720 rows, 320 -> 1280 columns
        x_model = 160 + (bottom_x - 160) * (y_model - 50) / (159 - 50)
        return (x_model + 0.5) * 4.0 - 0.5

    ego = result.debug_output["ego_lane"]
    expected_centre = (full_res_x(120, ego.reference_y) + full_res_x(230, ego.reference_y)) / 2.0
    assert result.warnings == []
    assert result.detected_segment_count == 4
    assert ego.reliable is True
    assert result.left_lane is not None and result.right_lane is not None and result.lane_polygon is not None
    assert result.left_lane.x2 < 640 <= result.right_lane.x2  # the boundaries either side of the camera
    assert result.lane_center_x == pytest.approx(expected_centre, abs=4.0)
    assert result.vehicle_offset_pixels == pytest.approx(640.0 - expected_centre, abs=4.0)
    assert result.vehicle_offset_pixels < 0  # lane centre (~691 px) right of the camera: vehicle left of the lane centre


def test_image_api_reports_the_model_that_served_the_request(tmp_path, monkeypatch) -> None:
    detector = get_ai_detector(_config(_checkpoint(tmp_path)))
    monkeypatch.setattr("app.services.detection_service.get_detector", lambda mode: detector)
    success, encoded = cv2.imencode(".png", np.full((72, 128, 3), 90, dtype=np.uint8))
    assert success

    response = TestClient(app).post("/api/detect/image", files={"file": ("road.png", encoded.tobytes(), "image/png")}, data={"algorithm": "ai"})

    assert response.status_code == 200
    metadata = response.json()["ai_metadata"]
    assert metadata["checkpoint_loaded"] is True
    assert metadata["device"] == "cpu"
    assert isinstance(metadata["lanes_detected"], int)
    assert isinstance(metadata["vehicle_offset_available"], bool)
