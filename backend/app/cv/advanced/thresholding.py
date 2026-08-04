from __future__ import annotations

import cv2
import numpy as np

from .config import AdvancedLaneDetectionConfig


def _normalize_channel(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _threshold_color(image: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask.astype(np.uint8)


def create_lane_binary(frame: np.ndarray, config: AdvancedLaneDetectionConfig) -> np.ndarray:
    """Create a binary lane mask from color and gradient cues."""
    if frame is None:
        return np.zeros((1, 1), dtype=np.uint8)
    array = np.asarray(frame)
    if array.ndim == 2:
        gray = array.astype(np.uint8)
        rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR) if array.dtype != np.uint8 else cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    else:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    white_mask = np.zeros_like(gray, dtype=np.uint8)
    if array.ndim != 2:
        white_mask = cv2.inRange(rgb, config.white_thresholds[0], config.white_thresholds[1])

    yellow_mask = np.zeros_like(gray, dtype=np.uint8)
    if array.ndim != 2:
        yellow_mask = _threshold_color(array, config.yellow_thresholds[0], config.yellow_thresholds[1])

    gradient_mask = apply_sobel(array, axis="x")
    combined = cv2.bitwise_or(white_mask, yellow_mask)
    combined = cv2.bitwise_or(combined, gradient_mask)
    return (combined > 0).astype(np.uint8)


def apply_sobel(frame: np.ndarray, axis: str = "x") -> np.ndarray:
    """Compute a Sobel-based gradient mask without crashing on constant images."""
    array = np.asarray(frame)
    if array.ndim == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    else:
        gray = array.astype(np.uint8)

    if gray.size == 0:
        return np.zeros(gray.shape[:2], dtype=np.uint8)

    sobel = cv2.Sobel(gray, cv2.CV_32F, 1 if axis == "x" else 0, 0 if axis == "x" else 1, ksize=3)
    abs_sobel = np.abs(sobel)
    max_value = float(abs_sobel.max()) if abs_sobel.size else 0.0
    if max_value <= 1e-6:
        return np.zeros(gray.shape[:2], dtype=np.uint8)

    normalized = np.clip(abs_sobel / max_value * 255.0, 0, 255).astype(np.uint8)
    return (normalized > 50).astype(np.uint8)
