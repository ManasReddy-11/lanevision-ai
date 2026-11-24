"""Every detector reports lane positions and the vehicle offset the same way (see app/cv/metrics/lane_metrics.py).

The test road is drawn in the bird's-eye view of the advanced detector (two straight lanes) and warped into the camera
image, so its geometry is known exactly. With the lanes either side of x = 600 in the bird's-eye view, the lane centre
is left of the camera reference point (the frame centre, x = 640): the vehicle is right of the lane centre, so the
offset must be positive, and mirroring the road must flip its sign.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from app.ai.adapter import AILaneDetectorAdapter
from app.ai.config import AIConfig
from app.ai.models import LaneSegmentationUNet
from app.cv.advanced.config import AdvancedLaneDetectionConfig
from app.cv.advanced.detector import AdvancedLaneDetector
from app.cv.advanced.perspective import get_perspective_matrices
from app.cv.detector import ClassicLaneDetector
from app.cv.metrics.lane_metrics import reference_row

WIDTH, HEIGHT = 1280, 720


def _road(lane_xs: tuple[int, int], mirror: bool = False) -> np.ndarray:
    _, Minv = get_perspective_matrices(WIDTH, HEIGHT, AdvancedLaneDetectionConfig())
    birds_eye = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for x in lane_xs:
        cv2.rectangle(birds_eye, (x - 10, 0), (x + 10, HEIGHT - 1), (255, 255, 255), -1)
    image = cv2.warpPerspective(birds_eye, Minv, (WIDTH, HEIGHT))
    return image[:, ::-1].copy() if mirror else image


def _ai_detection(image: np.ndarray, tmp_path):
    """The AI adapter's geometry on the road's lane pixels, standing in for the model's probability map."""
    torch.manual_seed(0)
    checkpoint = tmp_path / "unet.pth"
    torch.save({"state_dict": LaneSegmentationUNet(in_channels=3, num_classes=2).state_dict()}, checkpoint)
    adapter = AILaneDetectorAdapter(AIConfig(enabled=True, model_path=str(checkpoint), device="cpu", input_width=320, input_height=160))
    probability = cv2.resize((image[:, :, 0] > 127).astype(np.float32) * 0.9, (320, 160), interpolation=cv2.INTER_AREA)

    class FixedEngine:
        loader = adapter.engine.loader

        def run(self, frame):
            return {"probability_mask": probability, "mask": (probability >= 0.5).astype(np.uint8), "model_metadata": None}

    adapter.engine = FixedEngine()
    return adapter.detect(image)


@pytest.mark.parametrize("mirror", [False, True])
def test_all_detectors_share_the_offset_sign_and_lane_centre_convention(mirror, tmp_path) -> None:
    image = _road((300, 900), mirror=mirror)
    results = {
        "classic": ClassicLaneDetector().detect(image),
        "advanced": AdvancedLaneDetector().detect(image),
        "ai": _ai_detection(image, tmp_path),
    }
    sign = -1.0 if mirror else 1.0
    for name, result in results.items():
        assert result.vehicle_offset_pixels is not None, name
        assert result.frame_center_x == pytest.approx(WIDTH / 2.0), name
        # lane_center_x is a position in the frame, and the offset is frame centre minus lane centre
        assert result.vehicle_offset_pixels == pytest.approx(result.frame_center_x - result.lane_center_x), name
        assert sign * result.vehicle_offset_pixels > 15.0, name  # vehicle right of the lane centre (left when mirrored)
        assert sign * result.vehicle_offset_normalized > 0.0, name
        assert result.vehicle_offset_normalized == pytest.approx(result.vehicle_offset_pixels / result.lane_width_bottom_pixels), name
        assert result.left_lane is not None and result.right_lane is not None, name
        assert result.left_lane.x2 < result.lane_center_x < result.right_lane.x2, name
    offsets = [result.vehicle_offset_pixels for result in results.values()]
    assert max(offsets) - min(offsets) < 15.0  # the same road gives (nearly) the same offset whichever detector runs


def test_offsets_are_measured_at_the_shared_reference_row(tmp_path) -> None:
    image = _road((300, 900))
    for result in (ClassicLaneDetector().detect(image), AdvancedLaneDetector().detect(image)):
        assert result.left_lane.y2 == pytest.approx(reference_row(HEIGHT), abs=1.0) or result.left_lane.y2 > reference_row(HEIGHT)
    ego = _ai_detection(image, tmp_path).debug_output["ego_lane"]
    assert ego.reference_y == pytest.approx(reference_row(HEIGHT))


def test_detectors_report_missing_boundaries_instead_of_placeholders() -> None:
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for result in (ClassicLaneDetector().detect(blank), AdvancedLaneDetector().detect(blank)):
        assert result.left_lane is None and result.right_lane is None
        assert result.lane_center_x is None and result.vehicle_offset_pixels is None and result.vehicle_offset_normalized is None
        assert result.confidence == 0.0
        assert "No lane boundary was detected" in result.warnings
