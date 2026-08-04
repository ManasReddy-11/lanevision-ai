from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import UploadFile, status

from ..core.config import get_settings

settings = get_settings()

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
SUPPORTED_VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"}


class VideoValidationError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def validate_video_upload(file: UploadFile, temp_path: Path, size_bytes: int) -> dict[str, Any]:
    if file is None:
        raise VideoValidationError(status.HTTP_400_BAD_REQUEST, "No file provided")

    filename = file.filename or "video"
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise VideoValidationError(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported file extension")

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in SUPPORTED_VIDEO_MIME_TYPES:
        raise VideoValidationError(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported media type")

    max_bytes = settings.max_video_upload_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise VideoValidationError(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")

    if not temp_path.exists() or temp_path.stat().st_size <= 0:
        raise VideoValidationError(status.HTTP_400_BAD_REQUEST, "Empty upload")

    return {
        "filename": filename,
        "extension": extension,
        "content_type": content_type,
        "size_bytes": size_bytes,
    }
