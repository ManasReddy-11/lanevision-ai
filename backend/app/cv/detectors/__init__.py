from __future__ import annotations

from ...ai.adapter import AILaneDetectorAdapter
from ...ai.registry import get_ai_detector
from .advanced_adapter import AdvancedLaneDetectorAdapter
from .base import LaneDetector
from .classic_adapter import ClassicLaneDetectorAdapter


def get_detector(mode: str) -> LaneDetector:
    """Return a detector for the mode. Classic and advanced detectors are new instances (they may keep per-stream
    state); the AI detector is shared per process so its model is loaded once."""
    normalized = (mode or "classic").strip().lower()
    if normalized == "classic":
        return ClassicLaneDetectorAdapter()
    if normalized == "advanced":
        return AdvancedLaneDetectorAdapter()
    if normalized == "ai":
        return get_ai_detector()
    raise ValueError(f"Unsupported detector mode: {mode}")


__all__ = ["LaneDetector", "ClassicLaneDetectorAdapter", "AdvancedLaneDetectorAdapter", "AILaneDetectorAdapter", "get_detector"]
