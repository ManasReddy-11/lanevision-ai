from __future__ import annotations

import cv2
import numpy as np

from .config import AdvancedLaneDetectionConfig


def _normalize_points(points: tuple[tuple[float, float], ...], width: int, height: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for x_ratio, y_ratio in points:
        result.append((int(round(x_ratio * width)), int(round(y_ratio * height))))
    return result


def get_perspective_matrices(width: int, height: int, config: AdvancedLaneDetectionConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    config = config or AdvancedLaneDetectionConfig()
    src = np.array(_normalize_points(config.perspective_src_points, width, height), dtype=np.float32)
    dst = np.array(_normalize_points(config.perspective_dst_points, width, height), dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp_perspective(image: np.ndarray, M: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.warpPerspective(image, M, (width, height))


def unwarp_perspective(image: np.ndarray, Minv: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.warpPerspective(image, Minv, (width, height))
