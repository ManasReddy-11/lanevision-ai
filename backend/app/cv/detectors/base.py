from __future__ import annotations

from typing import Protocol

import numpy as np

from ..types import LaneDetectionResult


class LaneDetector(Protocol):
    def detect(self, frame: np.ndarray, debug: bool = False) -> LaneDetectionResult:
        ...
