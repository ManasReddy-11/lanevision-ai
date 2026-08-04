from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_image_bytes(ext: str = "png") -> bytes:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    image[:, :, 1] = 200
    image[:, :, 2] = 100
    success, encoded = cv2.imencode(f".{ext}", image)
    assert success
    return encoded.tobytes()


def test_valid_jpeg_upload_returns_detection_response() -> None:
    payload = _create_image_bytes("jpg")
    response = client.post(
        "/api/detect/image",
        files={"file": ("road.jpg", payload, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"]
    assert data["status"] == "success"
    assert data["algorithm"] == "classic"
    assert data["original"]["url"].startswith("/media/uploads/")
    assert data["result"]["url"].startswith("/media/processed/")
    assert data["metrics"]["confidence"] >= 0
    assert data["metrics"]["processing_time_ms"] >= 0
    assert data["metrics"]["detected_segments"] >= 0
    assert data["metrics"]["frame_center_x"] is not None
    assert data["lanes"]["left_detected"] in {True, False}
    assert data["lanes"]["right_detected"] in {True, False}
    assert isinstance(data["warnings"], list)


def test_valid_png_upload_and_processed_media_accessible() -> None:
    payload = _create_image_bytes("png")
    response = client.post(
        "/api/detect/image",
        files={"file": ("road.png", payload, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    processed_url = data["result"]["url"]
    media_response = client.get(processed_url)
    assert media_response.status_code == 200
    assert media_response.content


def test_unsupported_file_extension_is_rejected() -> None:
    response = client.post(
        "/api/detect/image",
        files={"file": ("road.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert "Unsupported" in response.json()["detail"]


def test_invalid_mime_type_is_rejected() -> None:
    response = client.post(
        "/api/detect/image",
        files={"file": ("road.jpg", b"not an image", "image/gif")},
    )

    assert response.status_code == 415
    assert "Unsupported" in response.json()["detail"]


def test_corrupted_image_is_rejected() -> None:
    response = client.post(
        "/api/detect/image",
        files={"file": ("corrupt.jpg", b"this is not a valid image", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "Invalid or corrupted image" in response.json()["detail"]


def test_oversized_upload_is_rejected() -> None:
    oversized_bytes = b"x" * (51 * 1024 * 1024)
    response = client.post(
        "/api/detect/image",
        files={"file": ("large.jpg", oversized_bytes, "image/jpeg")},
    )

    assert response.status_code == 413
    assert "large" in response.json()["detail"].lower() or "too large" in response.json()["detail"].lower()
