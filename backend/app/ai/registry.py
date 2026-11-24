"""One AI lane detector, and therefore one loaded model, per configuration per process.

Building an AILaneDetectorAdapter is cheap, but its ModelLoader reads the checkpoint on first use. Image requests and
every frame of a video therefore go through get_ai_detector so the checkpoint is read once, not once per call.
"""
from __future__ import annotations

import threading
from dataclasses import fields

from .adapter import AILaneDetectorAdapter
from .config import AIConfig

_lock = threading.Lock()
_detectors: dict[tuple, AILaneDetectorAdapter] = {}


def _cache_key(config: AIConfig) -> tuple:
    return tuple(getattr(config, field.name) for field in fields(config) if field.name != "metadata")


def get_ai_detector(config: AIConfig | None = None) -> AILaneDetectorAdapter:
    """Return the shared detector for this configuration (from the application settings when not given)."""
    config = config or AIConfig.from_settings()
    key = _cache_key(config)
    with _lock:
        detector = _detectors.get(key)
        if detector is None:
            detector = AILaneDetectorAdapter(config)
            _detectors[key] = detector
        return detector


def clear_ai_detectors() -> None:
    """Forget every shared detector (tests, or after changing AI settings in-process)."""
    with _lock:
        _detectors.clear()
