import cv2
import numpy as np

from app.cv.advanced.config import AdvancedLaneDetectionConfig
from app.cv.advanced.geometry import calculate_lane_center, calculate_lane_widths, calculate_polynomial_curvature
from app.cv.advanced.histogram import compute_histogram_lane_bases
from app.cv.advanced.perspective import get_perspective_matrices, warp_perspective
from app.cv.advanced.polynomial import PolynomialLane, fit_polynomial_lane, validate_polynomial_lane
from app.cv.advanced.sliding_windows import sliding_window_search
from app.cv.advanced.thresholding import apply_sobel, create_lane_binary
from app.cv.advanced.temporal import AdvancedTemporalState


def test_binary_threshold_output_is_binary_and_shape_preserved():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    binary = create_lane_binary(frame, AdvancedLaneDetectionConfig())
    assert binary.dtype == np.uint8
    assert binary.shape == frame.shape[:2]
    assert binary.max() <= 1


def test_sobel_gradient_handles_constant_images():
    frame = np.full((64, 64, 3), 50, dtype=np.uint8)
    sobel = apply_sobel(frame, axis="x")
    assert sobel.dtype == np.uint8
    assert sobel.shape == frame.shape[:2]


def test_perspective_matrices_and_warp_dimensions():
    width, height = 320, 240
    config = AdvancedLaneDetectionConfig()
    M, Minv = get_perspective_matrices(width, height, config)
    binary = np.zeros((height, width), dtype=np.uint8)
    warped = warp_perspective(binary, M, width, height)
    assert warped.shape == (height, width)
    assert M.shape == (3, 3)
    assert Minv.shape == (3, 3)


def test_histogram_lane_bases_find_left_and_right_candidates():
    binary = np.zeros((100, 100), dtype=np.uint8)
    binary[70:, 10:30] = 1
    binary[70:, 70:90] = 1
    left_base, right_base = compute_histogram_lane_bases(binary, split_ratio=0.5)
    assert left_base > 0
    assert right_base > 0


def test_sliding_window_search_extracts_pixels():
    binary = np.zeros((120, 160), dtype=np.uint8)
    binary[100:110, 35:45] = 1
    binary[100:110, 110:120] = 1
    left_x, right_x, left_y, right_y = sliding_window_search(binary, window_count=4, margin=20)
    assert len(left_x) > 0
    assert len(right_x) > 0


def test_polynomial_fit_and_validation():
    y = np.array([10, 20, 30, 40], dtype=np.float32)
    x = np.array([20, 18, 22, 25], dtype=np.float32)
    lane = fit_polynomial_lane(x, y, min_pixels=2)
    assert isinstance(lane, PolynomialLane)
    assert lane.pixel_count == 4
    assert np.isfinite(lane.coefficients).all()
    assert validate_polynomial_lane(lane, width=200, height=100)


def test_lane_width_and_center_convention():
    width, height = 200, 100
    left_x = 60.0
    right_x = 140.0
    widths = calculate_lane_widths(left_x, right_x, height)
    assert widths["width_bottom_pixels"] > 0
    center = calculate_lane_center(left_x, right_x, frame_center_x=width / 2.0)
    assert center == 0.0


def test_curvature_is_finite():
    coefficients = np.array([0.0001, -0.02, 60.0], dtype=np.float32)
    curvature = calculate_polynomial_curvature(coefficients)
    assert np.isfinite(curvature)


def test_advanced_temporal_state_resets():
    state = AdvancedTemporalState()
    state.update(np.array([0.1, 0.2, 0.3]), "left")
    state.reset()
    assert state.left_state is None
    assert state.right_state is None
