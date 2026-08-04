import cv2
import numpy as np

from app.cv.config import LaneDetectionConfig
from app.cv.detector import ClassicLaneDetector
from app.cv.types import LaneLine, LaneSegment
from app.cv.utils.validation import CVException, validate_frame
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
    assert vertices[0][1] > vertices[1][1]
    assert vertices[2][1] > vertices[3][1]


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
