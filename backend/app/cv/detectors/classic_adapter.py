from __future__ import annotations

import numpy as np

from ..config import LaneDetectionConfig
from ..detector import ClassicLaneDetector
from ..types import LaneDetectionResult
from .base import LaneDetector


class ClassicLaneDetectorAdapter:
    def __init__(self, config: LaneDetectionConfig | None = None) -> None:
        self._impl = ClassicLaneDetector(config=config or LaneDetectionConfig())

    def detect(self, frame: np.ndarray, debug: bool = False) -> LaneDetectionResult:
        return self._impl.detect(frame, debug=debug)
