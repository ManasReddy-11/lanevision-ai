from __future__ import annotations

import cv2
import numpy as np

from ..cv.types import LaneDetectionResult, LaneLine, LaneSegment


class LaneExtractor:
    def extract(self, mask: np.ndarray, original_frame: np.ndarray) -> tuple[LaneLine | None, LaneLine | None, dict[str, object]]:
        if mask is None:
            raise ValueError("Mask is required")
        if original_frame is None:
            raise ValueError("Original frame is required")

        height, width = mask.shape
        if height <= 0 or width <= 0:
            raise ValueError("Mask shape is invalid")

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, {"contours": 0}

        center_x = width / 2.0
        left_points = []
        right_points = []
        for contour in contours:
            if len(contour) < 3:
                continue
            xs = [point[0][0] for point in contour]
            ys = [point[0][1] for point in contour]
            if np.mean(xs) < center_x:
                left_points.extend(list(zip(xs, ys)))
            else:
                right_points.extend(list(zip(xs, ys)))

        def fit_lane(points: list[tuple[float, float]]) -> LaneLine | None:
            if len(points) < 3:
                return None
            points_array = np.array(points, dtype=np.float32)
            if len(points_array) < 3:
                return None
            x_values = points_array[:, 0]
            y_values = points_array[:, 1]
            coeffs = np.polyfit(y_values, x_values, 2)
            y_start = max(0, int(height * 0.6))
            y_end = height - 1
            x_start = float(np.polyval(coeffs, y_start))
            x_end = float(np.polyval(coeffs, y_end))
            return LaneLine(
                x1=max(0, min(x_start, width - 1)),
                y1=float(y_start),
                x2=max(0, min(x_end, width - 1)),
                y2=float(y_end),
                slope=float(coeffs[1]),
                confidence=0.5,
                support_segments=[LaneSegment(x1=float(x_start), y1=float(y_start), x2=float(x_end), y2=float(y_end))],
            )

        left_lane = fit_lane(left_points)
        right_lane = fit_lane(right_points)
        return left_lane, right_lane, {"contours": len(contours)}
