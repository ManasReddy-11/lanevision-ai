from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MediaReference(BaseModel):
    filename: str
    url: str
    width: Optional[int] = None
    height: Optional[int] = None


class DetectionResultReference(BaseModel):
    url: str


class DetectionMetrics(BaseModel):
    confidence: Optional[float] = None
    processing_time_ms: Optional[float] = None
    detected_segments: Optional[int] = None
    lane_center_x: Optional[float] = None
    frame_center_x: Optional[float] = None
    vehicle_offset_pixels: Optional[float] = None
    vehicle_offset_normalized: Optional[float] = None
    lane_width_bottom_pixels: Optional[float] = None
    lane_width_average_pixels: Optional[float] = None
    left_curvature_pixels: Optional[float] = None
    right_curvature_pixels: Optional[float] = None
    average_curvature_pixels: Optional[float] = None


class DetectionLanes(BaseModel):
    left_detected: bool
    right_detected: bool


class DetectionResponse(BaseModel):
    id: str
    status: str = Field(default="success")
    algorithm: str = Field(default="classic")
    original: MediaReference
    result: DetectionResultReference
    metrics: DetectionMetrics
    lanes: DetectionLanes
    warnings: list[str] = Field(default_factory=list)
