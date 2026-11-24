"""Lane-centre and vehicle-offset conventions shared by every detector (classic, advanced and AI).

- Positions (lane_center_x, frame_center_x, boundary x) are pixels from the left edge of the frame.
- Boundaries, lane centre and lane width are measured at one reference row near the bottom of the frame,
  OFFSET_REFERENCE_ROW_RATIO * (height - 1).
- vehicle_offset_pixels = frame_center_x - lane_center_x. Positive means the camera reference point (the vehicle) is
  right of the lane centre, negative means it is left of it.
- vehicle_offset_normalized = vehicle_offset_pixels / lane width at the reference row, so +/-0.5 puts the camera
  reference point on a lane boundary.
- Offsets are image pixels, not metres: converting them needs a camera calibration the system does not have.
"""
from __future__ import annotations

OFFSET_REFERENCE_ROW_RATIO = 0.9


def reference_row(height: int) -> float:
    """The image row at which lane boundaries, lane centre and vehicle offset are measured."""
    return OFFSET_REFERENCE_ROW_RATIO * (height - 1)


def line_x_at(x1: float, y1: float, x2: float, y2: float, y: float) -> float | None:
    """x of the straight line through (x1, y1) and (x2, y2) at row y; None for a horizontal line."""
    if abs(y2 - y1) < 1e-9:
        return None
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


def calculate_lane_center(left_x: float | None, right_x: float | None) -> float | None:
    """Return the midpoint between the left and right lane boundaries (a position, not an offset)."""
    if left_x is None or right_x is None:
        return None
    return (left_x + right_x) / 2.0


def calculate_vehicle_offset(frame_center_x: float | None, lane_center_x: float | None) -> float | None:
    """Return vehicle offset from lane center in pixels. Positive means the vehicle is right of the lane center."""
    if frame_center_x is None or lane_center_x is None:
        return None
    return frame_center_x - lane_center_x


def calculate_vehicle_offset_normalized(vehicle_offset_pixels: float | None, lane_width_pixels: float | None) -> float | None:
    """Return the offset as a fraction of the lane width (same sign as the pixel offset; +/-0.5 is on a boundary)."""
    if vehicle_offset_pixels is None or lane_width_pixels is None or lane_width_pixels <= 0:
        return None
    return vehicle_offset_pixels / lane_width_pixels


def calculate_confidence(
    supporting_segment_count: float,
    supporting_segment_length: float,
    left_lane_present: bool,
    right_lane_present: bool,
    slope_consistency: float,
) -> float:
    """Create a heuristic confidence score in the range [0, 1]."""
    base = 0.2
    segment_component = min(0.35, supporting_segment_count * 0.08)
    length_component = min(0.25, supporting_segment_length / 1000.0)
    lane_component = 0.1 if left_lane_present and right_lane_present else 0.05 if left_lane_present or right_lane_present else 0.0
    consistency_component = max(0.0, min(0.2, slope_consistency * 0.2))
    confidence = base + segment_component + length_component + lane_component + consistency_component
    return max(0.0, min(1.0, confidence))
