from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ai.lane_geometry import LaneGeometryConfig, extract_lanes, select_ego_lane

HEIGHT, WIDTH = 720, 1280


def _lane_map(lanes, dashed: set[int] = frozenset(), probability: float = 0.9) -> np.ndarray:
    """Draw lanes x = f(y) for y from 260 to the bottom as 6-px-wide bands of the given probability."""
    canvas = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    ys = np.arange(260, HEIGHT)
    for index, x_of_y in enumerate(lanes):
        points = np.stack([x_of_y(ys), ys], axis=1).round().astype(np.int32)
        if index in dashed:
            for start in range(0, len(points), 120):  # 60 px painted, 60 px gap, like a dashed marking
                cv2.polylines(canvas, [points[start:start + 60]], False, probability, thickness=6)
        else:
            cv2.polylines(canvas, [points], False, probability, thickness=6)
    return canvas


def _straight(bottom_x: float, vanishing_x: float = 640.0, vanishing_y: float = 240.0):
    return lambda y: vanishing_x + (bottom_x - vanishing_x) * (y - vanishing_y) / (HEIGHT - 1 - vanishing_y)


def test_extracts_every_lane_of_a_four_lane_highway() -> None:
    truth = [_straight(x) for x in (-150, 330, 950, 1450)]
    lanes = extract_lanes(_lane_map(truth))

    assert len(lanes) == 4
    for lane, x_of_y in zip(lanes, truth):
        for y in (400, 550, 700):
            if 0 <= x_of_y(y) < WIDTH:
                assert abs(lane.x_at(y) - x_of_y(y)) < 3.0


def test_dashed_and_curved_lanes_each_become_one_lane() -> None:
    curve = lambda y: 640.0 + 300.0 * ((y - 260.0) / 460.0) ** 2  # noqa: E731
    truth = [_straight(250), curve, _straight(1100)]
    lanes = extract_lanes(_lane_map(truth, dashed={0}))

    assert len(lanes) == 3
    assert lanes[0].y_bottom - lanes[0].y_top > 300  # the dashes were linked into one lane
    assert max(abs(lanes[1].x_at(y) - curve(y)) for y in range(300, 700, 20)) < 4.0


def test_noise_and_empty_maps_give_no_lanes() -> None:
    rng = np.random.default_rng(0)
    speckle = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    speckle[rng.integers(0, HEIGHT, 300), rng.integers(0, WIDTH, 300)] = 0.95
    assert extract_lanes(speckle) == []
    assert extract_lanes(np.zeros((HEIGHT, WIDTH), dtype=np.float32)) == []


def test_ego_lane_uses_the_boundaries_nearest_the_camera_and_signs_the_offset() -> None:
    lanes = extract_lanes(_lane_map([_straight(x) for x in (-150, 330, 950, 1450)]))
    ego = select_ego_lane(lanes, WIDTH, HEIGHT)

    reference_y = ego.reference_y
    expected_left, expected_right = _straight(330)(reference_y), _straight(950)(reference_y)
    assert ego.reliable is True
    assert abs(ego.left_x - expected_left) < 3.0
    assert abs(ego.right_x - expected_right) < 3.0
    assert abs(ego.offset_pixels - (640.0 - (expected_left + expected_right) / 2.0)) < 3.0
    assert ego.offset_pixels > 0  # lane centre is left of the camera: the vehicle sits right of the lane centre

    shifted = select_ego_lane(extract_lanes(_lane_map([_straight(x, vanishing_x=900) for x in (480, 1180)])), WIDTH, HEIGHT)
    assert shifted.reliable is True
    assert shifted.offset_pixels < 0  # lane centre right of the camera: vehicle left of the lane centre


@pytest.mark.parametrize(
    ("bottom_xs", "reason"),
    [
        ((900, 1300), "ego-lane left boundary not detected"),
        ((-400, 300), "ego-lane right boundary not detected"),
        ((600, 700), "implausible ego-lane width"),
    ],
)
def test_offset_is_unavailable_when_the_ego_lane_cannot_be_determined(bottom_xs, reason) -> None:
    lanes = extract_lanes(_lane_map([_straight(x) for x in bottom_xs]))
    ego = select_ego_lane(lanes, WIDTH, HEIGHT)

    assert ego.reliable is False
    assert ego.offset_pixels is None and ego.lane_center_x is None
    assert ego.reason == reason


def test_weak_lanes_do_not_produce_an_offset() -> None:
    lanes = extract_lanes(_lane_map([_straight(x) for x in (330, 950)], probability=0.55))
    ego = select_ego_lane(lanes, WIDTH, HEIGHT, LaneGeometryConfig(min_boundary_confidence=0.8))

    assert ego.reliable is False
    assert ego.reason == "low-confidence ego-lane boundary"


def test_lanes_seen_only_far_from_the_vehicle_are_not_used_as_ego_boundaries() -> None:
    far_only = _lane_map([_straight(330), _straight(950)])
    far_only[450:] = 0.0  # boundaries visible only in the upper part of the road
    ego = select_ego_lane(extract_lanes(far_only), WIDTH, HEIGHT)

    assert ego.reliable is False
    assert ego.reason == "no lane boundary detected near the vehicle"
