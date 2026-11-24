"""Lane instances from a lane-probability map, and the ego lane they imply.

The segmentation model outputs one lane probability per pixel. This module turns that map into individual lanes:
rows are scanned from the bottom of the image upwards, each row's above-threshold runs become candidate lane points,
points are linked into tracks by predicting each track's position from its recent slope (which bridges dashes and short
occlusions), fragments of the same lane are merged, and every track is fitted with a polynomial x = f(y). The ego lane
is the pair of lanes immediately left and right of the camera reference point at a reference row near the bottom.

Coordinates are pixels of the image the probability map covers: x grows to the right, y grows downwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LaneGeometryConfig:
    """Settings for lane extraction and ego-lane selection. Distances are fractions of the image size.

    probability_threshold and max_missed_rows were selected on the TuSimple validation split (label_data_0531, never
    the test split) with the baseline checkpoint: 0.3 and 12 rows scored best among thresholds {0.3, 0.4, 0.5} and
    gaps {6, 12}.
    """

    probability_threshold: float = 0.3
    row_step_ratio: float = 10.0 / 720.0  # rows are sampled every ~1.4% of the height (10 px at 720)
    top_ratio: float = 0.22  # rows above this fraction of the height are not scanned (sky and horizon)
    association_gate_ratio: float = 30.0 / 1280.0  # max distance between a track's predicted x and a point
    gate_growth_ratio: float = 10.0 / 1280.0  # extra distance allowed per row a track has gone unobserved
    max_missed_rows: int = 12  # rows a track may go unobserved (dashes, occlusion) before it is closed
    min_points: int = 4
    min_vertical_extent_ratio: float = 0.08
    polynomial_degree: int = 2
    max_lanes: int = 6
    reference_row_ratio: float = 0.9  # ego-lane boundaries and offset are measured at this row
    camera_center_ratio: float = 0.5  # camera reference x as a fraction of the width (camera mounted centrally)
    min_boundary_bottom_ratio: float = 0.55  # an ego boundary must be observed below this fraction of the height
    max_extrapolation_ratio: float = 0.25  # how far below its lowest observed point a boundary may be extended
    min_lane_width_ratio: float = 0.2  # plausible ego-lane width at the reference row, as a fraction of the width
    max_lane_width_ratio: float = 1.6
    min_boundary_confidence: float = 0.35


@dataclass(slots=True)
class LaneCurve:
    """One lane: observed points and a polynomial x = f(y) fitted to them."""

    coefficients: np.ndarray
    points: np.ndarray  # (N, 2) array of observed (x, y)
    mean_probability: float

    @property
    def y_top(self) -> float:
        return float(self.points[:, 1].min())

    @property
    def y_bottom(self) -> float:
        return float(self.points[:, 1].max())

    @property
    def support(self) -> int:
        return int(len(self.points))

    def x_at(self, y: float) -> float:
        """x of the lane at row y. Outside the observed rows the lane continues along its end tangent, so a curve
        fitted on the far field cannot swing wildly when extended towards the car."""
        y = float(y)
        if self.y_top <= y <= self.y_bottom:
            return float(np.polyval(self.coefficients, y))
        anchor = self.y_bottom if y > self.y_bottom else self.y_top
        slope = float(np.polyval(np.polyder(self.coefficients), anchor)) if len(self.coefficients) > 1 else 0.0
        return float(np.polyval(self.coefficients, anchor)) + slope * (y - anchor)

    def sample(self, ys, margin: float = 0.0) -> list[float | None]:
        """x at each row in ys that lies within the observed rows (+/- margin), otherwise None."""
        return [self.x_at(y) if self.y_top - margin <= y <= self.y_bottom + margin else None for y in ys]

    def polyline(self, step: float = 5.0) -> np.ndarray:
        ys = np.arange(self.y_top, self.y_bottom + 1e-6, step)
        return np.stack([np.polyval(self.coefficients, ys), ys], axis=1)

    def confidence(self, height: int, top_ratio: float) -> float:
        """Mean lane probability, scaled down when the lane covers less than half of the scanned road region."""
        road_rows = max((1.0 - top_ratio) * height, 1.0)
        coverage = min(1.0, (self.y_bottom - self.y_top) / (0.5 * road_rows))
        return float(self.mean_probability * coverage)


@dataclass(slots=True)
class EgoLane:
    """The lane the vehicle is in, measured at the reference row."""

    left: LaneCurve | None
    right: LaneCurve | None
    reference_y: float
    camera_x: float
    left_x: float | None
    right_x: float | None
    lane_center_x: float | None
    lane_width: float | None
    offset_pixels: float | None  # camera_x - lane_center_x; positive means the vehicle is right of the lane centre
    offset_normalized: float | None  # offset_pixels / lane_width
    confidence: float
    reliable: bool
    reason: str | None


class _Track:
    __slots__ = ("xs", "ys", "probabilities", "missed")

    def __init__(self, x: float, y: float, probability: float) -> None:
        self.xs = [x]
        self.ys = [y]
        self.probabilities = [probability]
        self.missed = 0

    def predict(self, y: float) -> float:
        recent = min(len(self.xs), 5)
        if recent >= 2 and self.ys[-1] != self.ys[-recent]:
            slope, intercept = np.polyfit(self.ys[-recent:], self.xs[-recent:], 1)
            return float(slope * y + intercept)
        return self.xs[-1]

    def extend(self, other: "_Track") -> None:
        self.xs += other.xs
        self.ys += other.ys
        self.probabilities += other.probabilities


def _row_points(profile: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    """Probability-weighted centre and peak probability of every above-threshold run in one image row."""
    above = profile >= threshold
    if not above.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above.astype(np.int8), [0]))))
    points = []
    for start, end in zip(edges[0::2], edges[1::2]):
        weights = profile[start:end].astype(np.float64)
        centre = float((np.arange(start, end) * weights).sum() / max(weights.sum(), 1e-12))
        points.append((centre, float(weights.max())))
    return points


def _merge_fragments(tracks: list[_Track], gate: float) -> list[_Track]:
    """Join a track with one that continues it further up the image (a lane split by a long dash gap or occlusion)."""
    merged = True
    while merged:
        merged = False
        best = None
        for lower in tracks:
            if len(lower.xs) < 2:
                continue
            for upper in tracks:
                if upper is lower or upper.ys[0] >= lower.ys[-1]:
                    continue  # upper must start above where lower ends (tracks run bottom to top)
                checks = upper.ys[: min(3, len(upper.ys))]
                residual = float(np.mean([abs(lower.predict(y) - x) for x, y in zip(upper.xs, checks)]))
                allowed = gate * (1.0 + (lower.ys[-1] - upper.ys[0]) / max(abs(lower.ys[0] - lower.ys[-1]), 1.0))
                if residual <= allowed and (best is None or residual < best[0]):
                    best = (residual, lower, upper)
        if best is not None:
            _, lower, upper = best
            lower.extend(upper)
            tracks.remove(upper)
            merged = True
    return tracks


def _fit(track: _Track, config: LaneGeometryConfig, width: int, height: int) -> LaneCurve | None:
    xs = np.asarray(track.xs, dtype=np.float64)
    ys = np.asarray(track.ys, dtype=np.float64)
    if len(xs) < config.min_points or ys.max() - ys.min() < config.min_vertical_extent_ratio * height:
        return None
    degree = config.polynomial_degree if len(xs) >= config.polynomial_degree + 4 else 1
    coefficients = np.polyfit(ys, xs, degree)
    residuals = np.abs(np.polyval(coefficients, ys) - xs)
    keep = residuals <= max(3.0 * float(np.median(residuals)), 0.006 * width)
    if config.min_points <= keep.sum() < len(xs):
        xs, ys = xs[keep], ys[keep]
        degree = config.polynomial_degree if len(xs) >= config.polynomial_degree + 4 else 1
        coefficients = np.polyfit(ys, xs, degree)
    probabilities = np.asarray(track.probabilities, dtype=np.float64)
    return LaneCurve(coefficients=coefficients, points=np.stack([xs, ys], axis=1), mean_probability=float(probabilities.mean()))


def extract_lanes(probability: np.ndarray, config: LaneGeometryConfig | None = None, rows=None) -> list[LaneCurve]:
    """Extract individual lanes from a 2-D lane-probability map, ordered left to right at the bottom of the image.

    rows: the image rows to scan (for example TuSimple h_samples); by default every row_step from the bottom up to
    top_ratio of the height.
    """
    config = config or LaneGeometryConfig()
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError("Expected a 2-D probability map")
    height, width = probability.shape
    if rows is None:
        step = max(1, int(round(config.row_step_ratio * height)))
        scan_rows = list(range(height - 1, int(config.top_ratio * height) - 1, -step))
    else:
        scan_rows = sorted({int(round(float(row))) for row in rows if 0 <= float(row) < height}, reverse=True)
    gate = config.association_gate_ratio * width
    growth = config.gate_growth_ratio * width

    active: list[_Track] = []
    closed: list[_Track] = []
    for y in scan_rows:
        points = _row_points(probability[y], config.probability_threshold)
        predictions = [track.predict(y) for track in active]
        candidates = sorted((abs(prediction - point[0]), t, p) for t, prediction in enumerate(predictions) for p, point in enumerate(points))
        matched_tracks: set[int] = set()
        matched_points: set[int] = set()
        for distance, t, p in candidates:
            if t in matched_tracks or p in matched_points or distance > gate + growth * active[t].missed:
                continue
            matched_tracks.add(t)
            matched_points.add(p)
            track = active[t]
            track.xs.append(points[p][0])
            track.ys.append(y)
            track.probabilities.append(points[p][1])
            track.missed = 0
        still_active = []
        for t, track in enumerate(active):
            if t not in matched_tracks:
                track.missed += 1
            (still_active if track.missed <= config.max_missed_rows else closed).append(track)
        active = still_active + [_Track(point[0], y, point[1]) for p, point in enumerate(points) if p not in matched_points]

    tracks = _merge_fragments(closed + active, gate)
    lanes = [lane for lane in (_fit(track, config, width, height) for track in tracks) if lane is not None]
    lanes.sort(key=lambda lane: lane.support * lane.mean_probability, reverse=True)
    lanes = lanes[: config.max_lanes]
    lanes.sort(key=lambda lane: lane.x_at(height - 1))
    return lanes


def select_ego_lane(lanes: list[LaneCurve], width: int, height: int, config: LaneGeometryConfig | None = None) -> EgoLane:
    """Pick the lanes immediately left and right of the camera reference point and measure the vehicle offset.

    The offset is reported only when both boundaries are observed near the vehicle, are confident enough, and form a
    lane of plausible width; otherwise it is None and `reason` says why.
    """
    config = config or LaneGeometryConfig()
    reference_y = config.reference_row_ratio * (height - 1)
    camera_x = config.camera_center_ratio * width
    anchored = []
    for lane in lanes:
        if lane.y_bottom < config.min_boundary_bottom_ratio * height:
            continue
        if reference_y - lane.y_bottom > config.max_extrapolation_ratio * height:
            continue
        anchored.append((lane.x_at(reference_y), lane))
    left = max((item for item in anchored if item[0] < camera_x), key=lambda item: item[0], default=None)
    right = min((item for item in anchored if item[0] >= camera_x), key=lambda item: item[0], default=None)

    boundary_confidence = [item[1].confidence(height, config.top_ratio) if item else 0.0 for item in (left, right)]
    confidence = float(sum(boundary_confidence) / 2.0)
    result = EgoLane(left=left[1] if left else None, right=right[1] if right else None, reference_y=reference_y, camera_x=camera_x,
                     left_x=left[0] if left else None, right_x=right[0] if right else None, lane_center_x=None, lane_width=None,
                     offset_pixels=None, offset_normalized=None, confidence=confidence, reliable=False, reason=None)
    if left is None and right is None:
        result.reason = "no lane boundary detected near the vehicle"
        return result
    if left is None or right is None:
        result.reason = f"ego-lane {'left' if left is None else 'right'} boundary not detected"
        return result
    width_px = right[0] - left[0]
    result.lane_width = float(width_px)
    if not config.min_lane_width_ratio * width <= width_px <= config.max_lane_width_ratio * width:
        result.reason = "implausible ego-lane width"
        return result
    if min(boundary_confidence) < config.min_boundary_confidence:
        result.reason = "low-confidence ego-lane boundary"
        return result
    result.lane_center_x = float((left[0] + right[0]) / 2.0)
    result.offset_pixels = float(camera_x - result.lane_center_x)
    result.offset_normalized = float(result.offset_pixels / width_px)
    result.reliable = True
    return result
