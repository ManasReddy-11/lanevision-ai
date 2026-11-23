from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_media_path(value: str, default: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = Field(default="LaneVision AI API")
    app_version: str = Field(default="1.0.0")
    environment: Literal["development", "testing", "production"] = Field(default="development")
    debug: bool = Field(default=True)
    api_prefix: str = Field(default="/api")
    frontend_url: str = Field(default="http://localhost:5173,http://127.0.0.1:5173")
    database_url: str = Field(default="sqlite:///./lanevision.db")
    upload_dir: str = Field(default_factory=lambda: str(PROJECT_ROOT / "uploads"))
    processed_dir: str = Field(default_factory=lambda: str(PROJECT_ROOT / "processed"))
    max_upload_size_mb: int = Field(default=50)
    max_video_upload_size_mb: int = Field(default=100)
    temporal_smoothing_alpha: float = Field(default=0.35)
    max_missing_lane_frames: int = Field(default=2)
    geometry_jump_threshold: float = Field(default=250.0)
    ai_enabled: bool = Field(default=True)
    ai_model_path: str | None = Field(default=None)
    ai_device: str = Field(default="auto")
    ai_input_width: int = Field(default=320)
    ai_input_height: int = Field(default=160)
    ai_mask_threshold: float = Field(default=0.5)
    ai_min_component_area: int = Field(default=64)
    ai_overlay_alpha: float = Field(default=0.35)
    ai_warmup: bool = Field(default=True)
    ai_model_name: str = Field(default="lane_segmentation_unet")
    ai_architecture: str = Field(default="u-net")
    ai_lane_class_id: int = Field(default=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def resolve_media_paths(self) -> "Settings":
        self.upload_dir = _resolve_media_path(self.upload_dir, PROJECT_ROOT / "uploads")
        self.processed_dir = _resolve_media_path(self.processed_dir, PROJECT_ROOT / "processed")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
