from __future__ import annotations

import numpy as np


class AdvancedTemporalState:
    """Simple EMA state for advanced detector polynomial coefficients."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self.left_state: np.ndarray | None = None
        self.right_state: np.ndarray | None = None

    def update(self, coefficients: np.ndarray, side: str) -> np.ndarray | None:
        array = np.asarray(coefficients, dtype=np.float32)
        if side == "left":
            if self.left_state is None:
                self.left_state = array
            else:
                self.left_state = self.alpha * array + (1.0 - self.alpha) * self.left_state
            return self.left_state
        if side == "right":
            if self.right_state is None:
                self.right_state = array
            else:
                self.right_state = self.alpha * array + (1.0 - self.alpha) * self.right_state
            return self.right_state
        return None

    def reset(self) -> None:
        self.left_state = None
        self.right_state = None
