import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cv.advanced.config import AdvancedLaneDetectionConfig
from app.cv.advanced.detector import AdvancedLaneDetector
from app.cv.advanced.histogram import compute_histogram_lane_bases
from app.cv.advanced.perspective import get_perspective_matrices, warp_perspective
from app.cv.advanced.polynomial import fit_polynomial_lane, validate_polynomial_lane
from app.cv.advanced.sliding_windows import sliding_window_search
from app.cv.advanced.thresholding import create_lane_binary
from app.cv.advanced.geometry import calculate_lane_center, calculate_lane_widths, calculate_polynomial_curvature

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "sample_data" / "images" / "curved_road.png"
if not IMAGE_PATH.exists():
    IMAGE_PATH = BASE_DIR.parent / "uploads" / "8a8d4440-ef4e-4289-9615-39761db00691.png"
if not IMAGE_PATH.exists():
    raise SystemExit("Synthetic curved image not found")

frame = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
if frame is None:
    raise SystemExit("Failed to read image")

config = AdvancedLaneDetectionConfig()
width, height = frame.shape[1], frame.shape[0]

save_dir = BASE_DIR / "processed" / "audit_debug"
save_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(save_dir / "original.png"), frame)

binary = create_lane_binary(frame, config)
cv2.imwrite(str(save_dir / "binary.png"), binary * 255)

M, Minv = get_perspective_matrices(width, height, config)
warped_binary = warp_perspective(binary.astype(np.uint8), M, width, height)
cv2.imwrite(str(save_dir / "birdseye.png"), warped_binary * 255)

histogram = np.sum(warped_binary[height // 2 :, :], axis=0)
hist_vis = np.zeros((height, width), dtype=np.uint8)
for idx, value in enumerate(histogram):
    hist_vis[:, idx] = 0
    if value > 0:
        hist_vis[height - int(min(value, height)) :, idx] = 255
cv2.imwrite(str(save_dir / "histogram.png"), hist_vis)

left_x, left_y, right_x, right_y = sliding_window_search(warped_binary, window_count=config.sliding_window_count, margin=config.window_margin)
left_lane_poly = fit_polynomial_lane(left_x, left_y, min_pixels=config.min_pixels_fit) if len(left_x) >= config.min_pixels_fit else None
right_lane_poly = fit_polynomial_lane(right_x, right_y, min_pixels=config.min_pixels_fit) if len(right_x) >= config.min_pixels_fit else None

poly_img = frame.copy()
if left_lane_poly is not None and validate_polynomial_lane(left_lane_poly, width, height):
    y_vals = np.linspace(height - 1, 0, 20).astype(np.int32)
    x_vals = np.polyval(left_lane_poly.coefficients, y_vals)
    for y, x in zip(y_vals, x_vals):
        cv2.circle(poly_img, (int(round(x)), int(round(y))), 2, (255, 0, 0), -1)
if right_lane_poly is not None and validate_polynomial_lane(right_lane_poly, width, height):
    y_vals = np.linspace(height - 1, 0, 20).astype(np.int32)
    x_vals = np.polyval(right_lane_poly.coefficients, y_vals)
    for y, x in zip(y_vals, x_vals):
        cv2.circle(poly_img, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
cv2.imwrite(str(save_dir / "polynomial_fit.png"), poly_img)

overlay = frame.copy()
if left_lane_poly is not None and right_lane_poly is not None and validate_polynomial_lane(left_lane_poly, width, height) and validate_polynomial_lane(right_lane_poly, width, height):
    y_values = np.linspace(height - 1, 0, 20).astype(np.int32)
    left_x_vals = np.polyval(left_lane_poly.coefficients, y_values)
    right_x_vals = np.polyval(right_lane_poly.coefficients, y_values)
    points = []
    for y, x in zip(y_values, left_x_vals):
        points.append([int(round(x)), int(round(y))])
    for y, x in zip(reversed(y_values), reversed(right_x_vals)):
        points.append([int(round(x)), int(round(y))])
    points_arr = np.array(points, dtype=np.int32)
    cv2.polylines(overlay, [points_arr], True, (0, 255, 0), 3)
cv2.imwrite(str(save_dir / "inverse_perspective_overlay.png"), overlay)

metrics = {}
if left_lane_poly is not None and validate_polynomial_lane(left_lane_poly, width, height):
    left_bottom_x = float(np.polyval(left_lane_poly.coefficients, height - 1))
    metrics['left_bottom_x'] = left_bottom_x
    metrics['left_curvature'] = calculate_polynomial_curvature(left_lane_poly.coefficients)
if right_lane_poly is not None and validate_polynomial_lane(right_lane_poly, width, height):
    right_bottom_x = float(np.polyval(right_lane_poly.coefficients, height - 1))
    metrics['right_bottom_x'] = right_bottom_x
    metrics['right_curvature'] = calculate_polynomial_curvature(right_lane_poly.coefficients)
if 'left_bottom_x' in metrics and 'right_bottom_x' in metrics:
    metrics['lane_width'] = abs(metrics['right_bottom_x'] - metrics['left_bottom_x'])
    metrics['lane_center'] = calculate_lane_center(metrics['left_bottom_x'], metrics['right_bottom_x'], frame_center_x=width / 2.0)
    metrics['vehicle_offset'] = metrics['lane_center']
    metrics['average_curvature'] = float(np.nanmean([metrics['left_curvature'], metrics['right_curvature']]))
metrics['left_lane_pixel_count'] = len(left_x)
metrics['right_lane_pixel_count'] = len(right_x)

print(json.dumps(metrics, indent=2))
print('saved_debug_artifacts', sorted(p.name for p in save_dir.glob('*')))
