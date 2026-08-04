from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..core.config import get_settings

settings = get_settings()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails server-side validation."""

    def __init__(self, detail: str, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def validate_upload(file_name: str, content_type: str | None, contents: bytes, max_size_mb: int | None = None) -> dict[str, Any]:
    """Validate uploaded file metadata, size, and extension before processing."""
    if not file_name or not file_name.strip():
        raise UploadValidationError("Empty filename", 400)

    if not contents:
        raise UploadValidationError("Empty upload", 400)

    max_bytes = (max_size_mb or settings.max_upload_size_mb) * 1024 * 1024
    if len(contents) > max_bytes:
        raise UploadValidationError("File too large", 413)

    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadValidationError("Unsupported file extension", 415)

    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        raise UploadValidationError("Unsupported media type", 415)

    return {"extension": suffix, "content_type": content_type or "application/octet-stream"}
