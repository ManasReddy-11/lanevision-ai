from __future__ import annotations

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.ai.config import AIConfig
from app.ai.model_loader import ModelLoader
from app.ai.postprocessing import MaskPostprocessor
from app.ai.preprocessing import Preprocessor
from app.cv.detectors import get_detector
from app.main import app

client = TestClient(app)


def test_ai_detector_returns_warning_when_model_is_unavailable() -> None:
    detector = get_detector("ai")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detector.detect(frame)

    assert result.success is True
    assert result.processed_frame is not None
    assert result.confidence >= 0.0
    assert any("AI model" in warning for warning in result.warnings)


def test_detection_api_accepts_ai_algorithm() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success

    response = client.post(
        "/api/detect/image",
        files={"file": ("road.png", encoded.tobytes(), "image/png")},
        data={"algorithm": "ai"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "ai"
    assert data["warnings"]
    assert data["ai_metadata"] is not None
    assert data["ai_metadata"]["model_name"]


def test_preprocessor_returns_tensor_with_batch_dimension() -> None:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    config = AIConfig(input_width=32, input_height=32)
    preprocessor = Preprocessor(config)
    tensor, original_size = preprocessor.prepare(frame, torch.device("cpu"))

    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    assert tensor.shape[2] == 32
    assert tensor.shape[3] == 32
    assert original_size == (96, 64)


def test_postprocessor_removes_small_components() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:8, 5:8] = 1
    config = AIConfig(min_component_area=10)
    postprocessor = MaskPostprocessor(config)
    result = postprocessor.process(mask, (20, 20))

    assert result.shape == (20, 20)
    assert result.sum() == 0


def test_model_loader_reports_missing_checkpoint() -> None:
    config = AIConfig(enabled=True, model_path="/tmp/does-not-exist.pth")
    loader = ModelLoader(config)

    try:
        loader.load()
    except Exception as exc:
        assert "not found" in str(exc).lower() or "configured" in str(exc).lower()
