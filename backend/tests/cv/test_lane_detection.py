import cv2
import numpy as np
import pytest

from app.cv.config import LaneDetectionConfig
from app.cv.detector import ClassicLaneDetector
from app.cv.types import LaneLine, LaneSegment
from app.cv.utils.validation import CVException, validate_frame
from app.cv.stages.hough import detect_hough_lines
from app.cv.stages.lane_fitting import fit_lane_line, classify_lane_segments
from app.cv.stages.roi import create_roi_mask, get_roi_vertices
from app.cv.stages.preprocessing import resize_if_required
from app.cv.metrics.lane_metrics import calculate_lane_center, calculate_vehicle_offset, calculate_confidence


def _make_synthetic_road_image() -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:] = (20, 20, 20)

    cv2.line(image, (280, 720), (420, 480), (255, 255, 255), 18)
    cv2.line(image, (900, 720), (760, 480), (255, 255, 255), 18)
    cv2.line(image, (320, 720), (470, 500), (255, 255, 255), 16)
    cv2.line(image, (920, 720), (760, 510), (255, 255, 255), 16)

    cv2.line(image, (320, 560), (470, 500), (0, 255, 255), 12)
    cv2.line(image, (860, 560), (760, 500), (0, 255, 255), 12)
    return image


def test_validate_frame_accepts_valid_array() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    validated = validate_frame(frame)
    assert validated.shape == (100, 200, 3)


def test_validate_frame_rejects_invalid_input() -> None:
    try:
        validate_frame(None)
    except CVException as exc:
        assert "frame" in str(exc).lower()
    else:
        raise AssertionError("Expected a CVException for invalid frame")


def test_roi_vertices_are_resolution_independent() -> None:
    width, height = 1280, 720
    vertices = get_roi_vertices(width, height, LaneDetectionConfig())
    assert len(vertices) == 4
    # bottom-left, top-left, top-right, bottom-right: the outline of the trapezoid
    assert vertices[0][1] > vertices[1][1]
    assert vertices[3][1] > vertices[2][1]
    assert vertices[0][0] < vertices[1][0] < vertices[2][0] < vertices[3][0]


def test_roi_mask_covers_the_whole_configured_trapezoid() -> None:
    # Regression: the vertices used to be ordered bottom-left, top-left, bottom-right, top-right, a self-crossing
    # "bow tie" that masked out the road in front of the vehicle (5.7% of a 1280x720 frame instead of 17.5%).
    config = LaneDetectionConfig()
    (bl_x, bottom), (tl_x, top), (tr_x, _), (br_x, _) = config.roi_polygon
    mask = create_roi_mask(1280, 720, config)
    expected = ((br_x - bl_x) + (tr_x - tl_x)) / 2.0 * (bottom - top)
    assert (mask > 0).mean() == pytest.approx(expected, rel=0.02)
    assert mask[int(0.9 * 720), 640] == 255  # the lane in front of the vehicle is inside the ROI


@pytest.mark.parametrize("shape", [(-1, 1, 4), (-1, 4)], ids=["opencv4", "opencv5"])
def test_hough_segments_are_parsed_for_each_opencv_output_shape(monkeypatch, shape) -> None:
    lines = np.array([[100, 700, 300, 400], [1000, 700, 800, 400]], dtype=np.int32).reshape(shape)
    monkeypatch.setattr("app.cv.stages.hough.cv2.HoughLinesP", lambda *args, **kwargs: lines)
    segments = detect_hough_lines(np.zeros((720, 1280), dtype=np.uint8) + 1, LaneDetectionConfig())
    assert [(s.x1, s.y1, s.x2, s.y2) for s in segments] == [(100, 700, 300, 400), (1000, 700, 800, 400)]


def test_roi_mask_dimensions_match_frame() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    mask = create_roi_mask(frame.shape[1], frame.shape[0], LaneDetectionConfig())
    assert mask.shape == (240, 320)
    assert mask.dtype == np.uint8


def test_slope_calculation_is_correct() -> None:
    segment = LaneSegment(x1=0, y1=100, x2=100, y2=0)
    assert abs(segment.slope) > 0


def test_segment_length_is_computed() -> None:
    segment = LaneSegment(x1=0, y1=0, x2=3, y2=4)
    assert segment.length == 5.0


def test_horizontal_line_is_rejected() -> None:
    segment = LaneSegment(x1=0, y1=10, x2=100, y2=10)
    assert segment.length > 0
    assert abs(segment.slope) < 1e-6


def test_left_and_right_lane_classification() -> None:
    left_segments = [LaneSegment(x1=100, y1=100, x2=200, y2=50, slope=1.0)]
    right_segments = [LaneSegment(x1=300, y1=100, x2=200, y2=50, slope=-1.0)]

    left, right = classify_lane_segments(left_segments, right_segments, width=640, height=480)
    assert left is not None
    assert right is not None


def test_weighted_lane_fitting_returns_lane_line() -> None:
    segments = [
        LaneSegment(x1=100, y1=400, x2=150, y2=300, slope=1.0, length=100.0),
        LaneSegment(x1=110, y1=400, x2=155, y2=300, slope=1.1, length=90.0),
    ]
    lane = fit_lane_line(segments, width=640, height=480, roi_top_y=180, roi_bottom_y=430)
    assert lane is not None
    assert isinstance(lane, LaneLine)


def test_image_bound_clipping_keeps_coordinates_in_bounds() -> None:
    lane = LaneLine(x1=1000, y1=1000, x2=-50, y2=-50, slope=-1.0, confidence=0.8)
    clipped = lane.clip_to_bounds(640, 480)
    assert clipped.x1 >= 0
    assert clipped.x2 <= 639
    assert clipped.y1 >= 0
    assert clipped.y2 <= 479


def test_lane_center_and_vehicle_offset_are_calculated() -> None:
    lane_center = calculate_lane_center(100, 300)
    offset = calculate_vehicle_offset(frame_center_x=320, lane_center_x=100)
    assert lane_center == 200.0
    assert offset == 220.0


def test_confidence_is_clamped_between_zero_and_one() -> None:
    confidence = calculate_confidence(2.0, 2.0, left_lane_present=True, right_lane_present=True, slope_consistency=0.9)
    assert 0.0 <= confidence <= 1.0


def test_detector_handles_no_lines() -> None:
    detector = ClassicLaneDetector()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result.success is True
    assert result.left_lane is None
    assert result.right_lane is None
    assert result.processed_frame.shape == frame.shape


def test_detector_handles_synthetic_road_image() -> None:
    detector = ClassicLaneDetector()
    frame = _make_synthetic_road_image()
    result = detector.detect(frame)
    assert result.success is True
    assert result.processed_frame.shape == frame.shape
    assert result.confidence >= 0.0
    # Regression (Linux, OpenCV 4): the best Hough segment of each side was passed on as the lane, and drawing it
    # failed with "'LaneSegment' object has no attribute 'clip_to_bounds'".
    assert isinstance(result.left_lane, LaneLine) and isinstance(result.right_lane, LaneLine)
    assert result.left_lane.x2 < 640 < result.right_lane.x2
    assert result.left_lane.y2 > result.left_lane.y1  # (x2, y2) is the end nearest the vehicle
    assert result.vehicle_offset_pixels is not None and result.vehicle_offset_pixels > 0  # lanes drawn left of centre


def test_detector_does_not_build_a_missing_boundary_from_the_other_side() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(frame, (900, 700), (720, 440), (255, 255, 255), 14)  # right boundary only
    result = ClassicLaneDetector().detect(frame)
    assert result.left_lane is None
    assert result.right_lane is not None and result.right_lane.x2 > 640
    assert result.vehicle_offset_pixels is None and result.lane_center_x is None
    assert "Only one lane boundary was detected" in result.warnings


def test_detector_reports_geometry_in_input_frame_coordinates() -> None:
    # Frames larger than the processing size are downscaled internally; output and positions stay in input pixels.
    small = _make_synthetic_road_image()
    large = cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    small_result = ClassicLaneDetector().detect(small)
    large_result = ClassicLaneDetector().detect(large)
    assert large_result.processed_frame.shape == large.shape
    assert large_result.frame_center_x == pytest.approx(960.0)
    assert large_result.vehicle_offset_pixels == pytest.approx(1.5 * small_result.vehicle_offset_pixels, rel=0.1)
