from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class LaneState:
    left: Optional[tuple[float, float, float, float]] = None
    right: Optional[tuple[float, float, float, float]] = None


class TemporalLaneSmoother:
    """Smooth lane geometry across adjacent frames with conservative persistence."""

    def __init__(self, alpha: float = 0.35, max_missing_frames: int = 3, geometry_jump_threshold: float = 250.0) -> None:
        self.alpha = alpha
        self.max_missing_frames = max_missing_frames
        self.geometry_jump_threshold = geometry_jump_threshold
        self.reset()

    def reset(self) -> None:
        self.previous_state = LaneState()
        self.missing_left_frames = 0
        self.missing_right_frames = 0

    def _is_large_jump(self, previous: Optional[tuple[float, float, float, float]], current: Optional[tuple[float, float, float, float]]) -> bool:
        if previous is None or current is None:
            return False
        prev_mid_y = (previous[1] + previous[3]) / 2.0
        curr_mid_y = (current[1] + current[3]) / 2.0
        prev_mid_x = (previous[0] + previous[2]) / 2.0
        curr_mid_x = (current[0] + current[2]) / 2.0
        return abs(curr_mid_x - prev_mid_x) > self.geometry_jump_threshold or abs(curr_mid_y - prev_mid_y) > self.geometry_jump_threshold

    def _smooth_lane(self, previous: Optional[tuple[float, float, float, float]], current: Optional[tuple[float, float, float, float]]) -> Optional[tuple[float, float, float, float]]:
        if current is None:
            return None
        if previous is None:
            return current
        if self._is_large_jump(previous, current):
            return current
        return tuple(
            self.alpha * current[idx] + (1.0 - self.alpha) * previous[idx]
            for idx in range(4)
        )

    def update(self, left: Optional[tuple[float, float, float, float]], right: Optional[tuple[float, float, float, float]]) -> LaneState:
        smoothed_left = self.previous_state.left
        smoothed_right = self.previous_state.right

        if left is None:
            self.missing_left_frames += 1
            if self.missing_left_frames < self.max_missing_frames and self.previous_state.left is not None:
                smoothed_left = self.previous_state.left
            else:
                smoothed_left = None
                self.previous_state.left = None
        else:
            self.missing_left_frames = 0
            smoothed_left = self._smooth_lane(self.previous_state.left, left)

        if right is None:
            self.missing_right_frames += 1
            if self.missing_right_frames < self.max_missing_frames and self.previous_state.right is not None:
                smoothed_right = self.previous_state.right
            else:
                smoothed_right = None
                self.previous_state.right = None
        else:
            self.missing_right_frames = 0
            smoothed_right = self._smooth_lane(self.previous_state.right, right)

        self.previous_state = LaneState(left=smoothed_left, right=smoothed_right)
        return self.previous_state
