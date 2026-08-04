import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cv.advanced.config import AdvancedLaneDetectionConfig
from app.cv.advanced.detector import AdvancedLaneDetector

image_path = Path(__file__).resolve().parent.parent / "sample_data" / "images" / "curved_road.png"
if not image_path.exists():
    image_path = Path(__file__).resolve().parent.parent.parent / "uploads" / "8a8d4440-ef4e-4289-9615-39761db00691.png"
if not image_path.exists():
    raise SystemExit("Synthetic curved image not found")

frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if frame is None:
    raise SystemExit("Failed to read image")

detector = AdvancedLaneDetector(AdvancedLaneDetectionConfig())
start = time.perf_counter()
result = detector.detect(frame, debug=True)
elapsed_ms = (time.perf_counter() - start) * 1000.0

def to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

values = {
    "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
    "processing_time_ms": round(float(result.processing_time_ms), 6),
    "confidence": round(float(result.confidence), 6),
    "left_lane_pixel_count": int(len(result.left_lane.support_segments) if result.left_lane is not None else 0),
    "right_lane_pixel_count": int(len(result.right_lane.support_segments) if result.right_lane is not None else 0),
    "lane_width": None if to_float(getattr(result, "lane_width_bottom_pixels", None)) is None else round(to_float(getattr(result, "lane_width_bottom_pixels", None)), 6),
    "lane_center": None if to_float(result.lane_center_x) is None else round(to_float(result.lane_center_x), 6),
    "vehicle_offset": None if to_float(result.vehicle_offset_pixels) is None else round(to_float(result.vehicle_offset_pixels), 6),
    "left_curvature": None if to_float(getattr(result, "left_curvature_pixels", None)) is None else round(to_float(getattr(result, "left_curvature_pixels", None)), 6),
    "right_curvature": None if to_float(getattr(result, "right_curvature_pixels", None)) is None else round(to_float(getattr(result, "right_curvature_pixels", None)), 6),
    "average_curvature": None if to_float(getattr(result, "average_curvature_pixels", None)) is None else round(to_float(getattr(result, "average_curvature_pixels", None)), 6),
    "debug_keys": list(result.debug_output.keys()) if result.debug_output else [],
}

print(json.dumps(values, indent=2))
finite_values = [
    values["processing_time_ms"],
    values["confidence"],
    values["lane_center"],
    values["vehicle_offset"],
    values["average_curvature"],
]
finite_values = [v for v in finite_values if v is not None]
print("finite", all(np.isfinite(finite_values)))
print("json_safe", json.dumps(values))
