from __future__ import annotations

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.ai.adapter import AILaneDetectorAdapter
from app.ai.config import AIConfig
from app.ai.inference import AIInferenceEngine
from app.ai.model_loader import ModelLoader
from app.ai.models import LaneSegmentationUNet
from app.ai.postprocessing import MaskPostprocessor
from app.ai.preprocessing import Preprocessor
from app.cv.detectors import get_detector
from app.main import app

client = TestClient(app)


def _write_unet_checkpoint(path, untrained_channel_bias: float = 0.0):
    torch.manual_seed(0)
    model = LaneSegmentationUNet(in_channels=3, num_classes=2)
    with torch.no_grad():
        model.out.bias[1] += untrained_channel_bias
    torch.save({"state_dict": model.state_dict()}, path)
    return path


def test_ai_detector_returns_warning_when_model_is_unavailable() -> None:
    detector = get_detector("ai")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detector.detect(frame)

    assert result.success is True
    assert result.processed_frame is not None
    assert result.confidence >= 0.0
    assert any("AI model" in warning for warning in result.warnings)


def test_detection_api_refuses_ai_when_no_checkpoint_is_configured() -> None:
    # Without AI_MODEL_PATH the request fails explicitly instead of returning the unprocessed image as a "success".
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success

    response = client.post(
        "/api/detect/image",
        files={"file": ("road.png", encoded.tobytes(), "image/png")},
        data={"algorithm": "ai"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "AI model unavailable: AI model weights are not configured. "
        "Set AI_MODEL_PATH to a trained checkpoint file (see backend/app/ai/README.md)."
    )


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


def test_inference_reads_the_lane_channel_and_activation_used_in_training(tmp_path) -> None:
    frame = np.random.default_rng(0).integers(0, 256, (32, 64, 3), dtype=np.uint8)
    tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    torch.manual_seed(0)
    model = LaneSegmentationUNet(in_channels=3, num_classes=2).eval()
    with torch.no_grad():
        # Centre channel 0 so roughly half the pixels are lane, and make the never-trained channel 1 dominant,
        # so reading channel 1 (for example through a softmax) cannot pass by accident.
        model.out.bias[0] -= model(tensor)[0, 0].median()
        model.out.bias[1] += 5.0
    checkpoint = tmp_path / "unet.pth"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    engine = AIInferenceEngine(AIConfig(enabled=True, model_path=str(checkpoint), device="cpu", input_width=64, input_height=32))

    result = engine.run(frame)

    # The prediction exactly as backend/training/metrics.py scores it during training: sigmoid(channel 0) >= 0.5.
    with torch.no_grad():
        logits = engine.loader.model(tensor)
    expected = (torch.sigmoid(logits[0, 0]) >= 0.5).numpy().astype(np.uint8)
    assert 0 < expected.sum() < expected.size
    assert np.array_equal(result["mask"], expected)


def test_default_ai_settings_describe_the_trained_unet_contract() -> None:
    config = AIConfig.from_settings()
    assert config.lane_class_id == 0
    assert config.architecture == "u-net"


def test_ai_adapter_handles_non_square_frames_with_a_loaded_checkpoint(tmp_path) -> None:
    checkpoint = _write_unet_checkpoint(tmp_path / "unet.pth")
    config = AIConfig(enabled=True, model_path=str(checkpoint), device="cpu", input_width=64, input_height=32, min_component_area=0)
    frame = np.random.default_rng(1).integers(0, 256, (72, 128, 3), dtype=np.uint8)

    result = AILaneDetectorAdapter(config).detect(frame)

    assert result.success is True
    assert not any(warning.startswith("AI model unavailable") for warning in result.warnings)
    assert result.processed_frame.shape == frame.shape
    assert result.debug_output["mask"].shape == frame.shape[:2]


def test_model_loader_status_reports_width_and_height_before_loading(tmp_path) -> None:
    checkpoint = _write_unet_checkpoint(tmp_path / "unet.pth")
    loader = ModelLoader(AIConfig(enabled=True, model_path=str(checkpoint), input_width=320, input_height=160))
    assert loader.get_status()["input_size"] == [320, 160]


def test_model_loader_reports_missing_checkpoint() -> None:
    config = AIConfig(enabled=True, model_path="/tmp/does-not-exist.pth")
    loader = ModelLoader(config)

    try:
        loader.load()
    except Exception as exc:
        assert "not found" in str(exc).lower() or "configured" in str(exc).lower()
