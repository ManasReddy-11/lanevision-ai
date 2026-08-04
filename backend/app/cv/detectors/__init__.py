from __future__ import annotations

from .advanced_adapter import AdvancedLaneDetectorAdapter
from .base import LaneDetector
from .classic_adapter import ClassicLaneDetectorAdapter


def get_detector(mode: str) -> LaneDetector:
    normalized = (mode or "classic").strip().lower()
    if normalized == "classic":
        return ClassicLaneDetectorAdapter()
    if normalized == "advanced":
        return AdvancedLaneDetectorAdapter()
    raise ValueError(f"Unsupported detector mode: {mode}")


__all__ = ["LaneDetector", "ClassicLaneDetectorAdapter", "AdvancedLaneDetectorAdapter", "get_detector"]
