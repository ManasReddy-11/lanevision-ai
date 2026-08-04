from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_video_bytes(tmp_path, ext: str = "mp4") -> bytes:
    path = tmp_path / f"sample.{ext}"
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    writer.write(frame)
    writer.release()
    return path.read_bytes()


def test_valid_video_upload_returns_video_response(tmp_path):
    payload = _create_video_bytes(tmp_path, "mp4")
    response = client.post(
        "/api/detect/video",
        files={"file": ("sample.mp4", payload, "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["algorithm"] == "classic"
    assert data["original"]["url"].startswith("/media/uploads/")
    assert data["result"]["url"].startswith("/media/processed/")
    assert data["metrics"]["frames_processed"] >= 1
    assert data["metrics"]["source_fps"] >= 0


def test_unsupported_video_extension_is_rejected():
    response = client.post(
        "/api/detect/video",
        files={"file": ("sample.txt", b"not a video", "text/plain")},
    )
    assert response.status_code == 415


def test_invalid_video_mime_is_rejected():
    response = client.post(
        "/api/detect/video",
        files={"file": ("sample.mp4", b"not a video", "image/png")},
    )
    assert response.status_code == 415


def test_oversized_video_is_rejected():
    response = client.post(
        "/api/detect/video",
        files={"file": ("sample.mp4", b"x" * (101 * 1024 * 1024), "video/mp4")},
    )
    assert response.status_code == 413
