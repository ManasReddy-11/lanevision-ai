from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.cv.temporal.smoother import TemporalLaneSmoother
from app.services.video_processing_service import VideoProcessingService


def test_temporal_smoother_initial_state_and_reset():
    smoother = TemporalLaneSmoother(alpha=0.5, max_missing_frames=2)
    assert smoother.update(None, None).left is None
    assert smoother.update(None, None).right is None
    smoother.reset()
    assert smoother.previous_state.left is None
    assert smoother.previous_state.right is None


def test_temporal_smoother_smooths_and_persists_missing_lanes():
    smoother = TemporalLaneSmoother(alpha=0.5, max_missing_frames=2)
    first = smoother.update((10, 100, 200, 600), (300, 100, 500, 600))
    assert first.left is not None
    assert first.right is not None

    second = smoother.update((12, 105, 205, 610), None)
    assert second.left is not None
    assert second.right is not None

    third = smoother.update(None, None)
    assert third.left is not None
    assert third.right is None


def test_video_processing_service_creates_output_video(tmp_path):
    service = VideoProcessingService()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    writer.write(frame)
    writer.release()

    class FakeUploadFile:
        def __init__(self, path):
            self.filename = path.name
            self.content_type = "video/mp4"
            self.file = open(path, "rb")

        def __del__(self):
            try:
                self.file.close()
            except Exception:
                pass

    upload = FakeUploadFile(video_path)
    response = service.process_upload(upload)
    assert response.status == "success"
    assert response.metrics.frames_processed >= 1
    assert response.result.url.startswith("/media/processed/")
    settings = get_settings()
    output_path = Path(settings.processed_dir) / Path(response.result.url.replace("/media/processed/", "")).name
    assert output_path.exists()