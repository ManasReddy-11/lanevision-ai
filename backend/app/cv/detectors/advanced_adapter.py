from __future__ import annotations

import numpy as np

from ..advanced.config import AdvancedLaneDetectionConfig
from ..advanced.detector import AdvancedLaneDetector
from ..types import LaneDetectionResult
from .base import LaneDetector


class AdvancedLaneDetectorAdapter:
    def __init__(self, config: AdvancedLaneDetectionConfig | None = None) -> None:
        self._impl = AdvancedLaneDetector(config=config or AdvancedLaneDetectionConfig())

    def detect(self, frame: np.ndarray, debug: bool = False) -> LaneDetectionResult:
        return self._impl.detect(frame, debug=debug)
