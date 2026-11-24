from __future__ import annotations

import cv2
import numpy as np

from .config import AIConfig


class OverlayRenderer:
    """Draw the segmentation mask, every detected lane, and the ego lane on a copy of the frame."""

    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig.from_settings()

    def render(self, frame: np.ndarray, mask: np.ndarray, lanes=None, ego=None, polygon: np.ndarray | None = None, held=frozenset()) -> np.ndarray:
        """held: ids (id()) of lanes reported from previous frames rather than detected in this one; drawn grey and dashed."""
        if frame is None:
            raise ValueError("Frame is required")
        if mask is None:
            raise ValueError("Mask is required")
        overlay = frame.copy()
        color_mask = np.zeros_like(overlay)
        color_mask[:, :, 1] = (mask * 255).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 1.0 - self.config.overlay_alpha, color_mask, self.config.overlay_alpha, 0)
        if polygon is not None:
            fill = overlay.copy()
            cv2.fillPoly(fill, [polygon], (255, 128, 0))
            overlay = cv2.addWeighted(overlay, 0.75, fill, 0.25, 0)
        for lane in lanes or []:
            points = lane.polyline().round().astype(np.int32)
            if len(points) >= 2:
                if id(lane) in held:
                    _dashed(overlay, points, (160, 160, 160), 2)
                else:
                    cv2.polylines(overlay, [points], False, (0, 215, 255), 2, cv2.LINE_AA)
        if ego is not None:
            for boundary in (ego.left, ego.right):
                if boundary is None:
                    continue
                ys = np.arange(boundary.y_top, ego.reference_y + 1e-6, 5.0)
                points = np.array([(boundary.x_at(y), y) for y in ys], dtype=np.float32).round().astype(np.int32)
                if len(points) >= 2:
                    if id(boundary) in held:
                        _dashed(overlay, points, (0, 200, 0), 4)
                    else:
                        cv2.polylines(overlay, [points], False, (0, 255, 0), 4, cv2.LINE_AA)
        return overlay


def _dashed(image: np.ndarray, points: np.ndarray, color, thickness: int) -> None:
    for index in range(0, len(points) - 1, 2):
        cv2.line(image, tuple(int(v) for v in points[index]), tuple(int(v) for v in points[index + 1]), color, thickness, cv2.LINE_AA)
