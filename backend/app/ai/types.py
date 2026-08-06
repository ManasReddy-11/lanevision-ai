from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AIResultMetadata:
    model_name: str
    architecture: str
    device: str
    checkpoint_loaded: bool
    input_width: int
    input_height: int
    threshold: float
    parameter_count: int
    inference_time_ms: float
    segmentation_coverage: float
