from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VideoMediaReference(BaseModel):
    filename: str
    url: str
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None


class VideoDetectionResultReference(BaseModel):
    url: str


class VideoDetectionMetrics(BaseModel):
    frames_processed: int
    source_fps: Optional[float] = None
    average_processing_fps: Optional[float] = None
    average_processing_time_ms: Optional[float] = None
    average_confidence: Optional[float] = None
    detection_success_rate: Optional[float] = None
    both_lane_detection_rate: Optional[float] = None
    average_vehicle_offset_pixels: Optional[float] = None


class VideoDetectionResponse(BaseModel):
    id: str
    status: str = Field(default="success")
    algorithm: str = Field(default="classic")
    original: VideoMediaReference
    result: VideoDetectionResultReference
    metrics: VideoDetectionMetrics
    warnings: list[str] = Field(default_factory=list)
    ai_metadata: Optional[dict] = None
