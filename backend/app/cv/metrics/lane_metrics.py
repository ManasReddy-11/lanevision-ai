from __future__ import annotations


def calculate_lane_center(left_x: float | None, right_x: float | None) -> float | None:
    """Return the midpoint between the left and right lane boundaries."""
    if left_x is None or right_x is None:
        return None
    return (left_x + right_x) / 2.0


def calculate_vehicle_offset(frame_center_x: float | None, lane_center_x: float | None) -> float | None:
    """Return vehicle offset from lane center in pixels. Positive means the vehicle is right of the lane center."""
    if frame_center_x is None or lane_center_x is None:
        return None
    return frame_center_x - lane_center_x


def calculate_vehicle_offset_normalized(frame_center_x: float | None, lane_center_x: float | None) -> float | None:
    """Return normalized offset based on frame center."""
    if frame_center_x is None or lane_center_x is None:
        return None
    offset = lane_center_x - frame_center_x
    return offset / max(frame_center_x, 1.0)


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
