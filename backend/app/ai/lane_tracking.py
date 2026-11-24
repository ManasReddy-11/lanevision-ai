"""Temporal lane tracking for video, and an experimental lane departure warning.

extract_lanes finds lanes in one frame. In a video the same lane should keep one identity, its position should not
jitter from frame to frame, a dash gap or a frame where a faint lane is missed should not make it vanish, and a single
spurious detection should not be shown. LaneTracker does this with a deliberately simple scheme:

- Every lane is a track: the lane's x position on a fixed grid of image rows.
- Detections are matched to tracks by the mean horizontal distance between them (greedy, closest first, gated).
- A matched track blends each observed row: x = smoothing * observed + (1 - smoothing) * previous. A row that is not
  observed in this frame (a dash gap, a short occlusion) keeps its position, moved by the lane's lateral shift, for up
  to row_memory_frames frames.
- A new lane is reported once it has been detected in min_hits consecutive frames.
- A lane that is not detected is held (reported with held=True and a decaying confidence) for up to max_missed_frames
  frames, then dropped. The ego-lane confidence uses that decayed value, so a boundary that is only being held soon
  stops supporting a reliable vehicle offset.

The tracker keeps per-video state: create one per video (or call reset()) so nothing carries over between videos.

LaneDepartureMonitor raises an experimental warning when the vehicle offset stays beyond a fraction of the lane width.
Offsets are image pixels relative to an assumed camera position (the frame centre), not calibrated distances; the
warning is a demonstration feature, not a driver-assistance or safety function.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lane_geometry import EgoLane, LaneCurve, LaneGeometryConfig


@dataclass(slots=True)
class LaneTrackerConfig:
    """Defaults chosen on the TuSimple validation clips (label_data_0531, 351 clips of 20 frames at 20 fps), never on
    the test split; see docs/temporal_tracking.md. Frame counts assume about 20 fps."""

    smoothing: float = 0.5  # weight of the newly observed position in the per-row moving average (1.0 = no smoothing)
    gate_ratio: float = 0.04  # max mean horizontal distance between a track and a detection, as a fraction of the width
    min_hits: int = 2  # consecutive frames a new lane must be detected before it is reported
    max_missed_frames: int = 10  # frames a lane is held without a matching detection before it is dropped
    row_memory_frames: int = 3  # frames an unobserved row of a matched lane keeps its (shifted) position
    held_confidence_decay: float = 0.7  # lane probability multiplier for every frame the lane is held
    min_rows: int = 4  # rows a lane needs to be fitted
    polynomial_degree: int = 2
    consistency_rules: bool = True  # drop remembered rows that disagree with the lane's current detection, and do
    #                                 not report a held lane that crosses an observed lane near the vehicle


@dataclass(slots=True)
class TrackedLane:
    """A lane as reported for the current frame."""

    track_id: int
    curve: LaneCurve
    hits: int  # frames in which the lane was detected
    missed: int  # consecutive frames without a detection (0: detected in this frame)

    @property
    def held(self) -> bool:
        return self.missed > 0


@dataclass(slots=True)
class TrackingEvents:
    """What changed in this frame."""

    appeared: list[int] = field(default_factory=list)  # track ids reported for the first time
    disappeared: list[int] = field(default_factory=list)  # reported track ids that were dropped
    ego_lane_change: str | None = None  # "left" or "right" when the vehicle moved into the neighbouring lane


class _Track:
    __slots__ = ("track_id", "x", "row_age", "probability", "hits", "missed", "confirmed", "state")

    def __init__(self, track_id: int, x: np.ndarray, probability: float) -> None:
        self.track_id = track_id
        self.x = x
        self.row_age = np.zeros(len(x), dtype=np.int32)
        self.probability = probability
        self.hits = 1
        self.missed = 0
        self.confirmed = False
        self.state: dict = {}  # extra per-track state for alternative filters


class LaneTracker:
    """Track lanes across the frames of one video. See the module docstring for the method."""

    def __init__(self, config: LaneTrackerConfig | None = None, geometry: LaneGeometryConfig | None = None, rows=None) -> None:
        self.config = config or LaneTrackerConfig()
        self.geometry = geometry or LaneGeometryConfig()
        self._fixed_rows = None if rows is None else np.asarray(sorted({float(row) for row in rows}), dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Forget every lane (start of a new video)."""
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._shape: tuple[int, int] | None = None
        self._rows: np.ndarray | None = None
        self._reported: set[int] = set()
        self._ego_ids: tuple[int | None, int | None] = (None, None)
        self.frames = 0

    def _grid(self, width: int, height: int) -> np.ndarray:
        if self._shape != (width, height):
            self.reset()
            self._shape = (width, height)
            if self._fixed_rows is not None:
                self._rows = self._fixed_rows
            else:
                step = max(1, int(round(self.geometry.row_step_ratio * height)))
                self._rows = np.arange(height - 1, int(self.geometry.top_ratio * height) - 1, -step, dtype=np.float64)[::-1]
        return self._rows

    def update(self, lanes: list[LaneCurve], width: int, height: int) -> tuple[list[TrackedLane], TrackingEvents]:
        """Associate this frame's detected lanes with the tracks and return the lanes to report, left to right."""
        rows = self._grid(width, height)
        self.frames += 1
        config = self.config
        detections = [np.array([np.nan if x is None else x for x in lane.sample(rows)], dtype=np.float64) for lane in lanes]
        gate = config.gate_ratio * width

        candidates = []
        for t, track in enumerate(self._tracks):
            for d, observed in enumerate(detections):
                distance = self._distance(track, observed)
                if distance is not None and distance <= gate:
                    candidates.append((distance, t, d))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, t, d in sorted(candidates):
            if t in matched_tracks or d in matched_detections:
                continue
            matched_tracks.add(t)
            matched_detections.add(d)
            self._absorb(self._tracks[t], detections[d], lanes[d].mean_probability)
            if config.consistency_rules:
                self._forget_inconsistent_rows(self._tracks[t], lanes[d], gate)

        survivors = []
        for t, track in enumerate(self._tracks):
            if t not in matched_tracks:
                track.missed += 1
                track.probability *= config.held_confidence_decay
                self._hold(track)
                if not track.confirmed or track.missed > config.max_missed_frames:
                    continue  # a tentative lane must be detected in consecutive frames; a held lane expires
            survivors.append(track)
        for d, observed in enumerate(detections):
            if d not in matched_detections and np.isfinite(observed).sum() >= config.min_rows:
                track = _Track(self._next_id, observed.copy(), lanes[d].mean_probability)
                self._next_id += 1
                survivors.append(track)
        self._tracks = survivors

        reported: list[TrackedLane] = []
        for track in self._tracks:
            if not track.confirmed and track.hits >= config.min_hits:
                track.confirmed = True
            if not track.confirmed:
                continue
            curve = self._curve(track, rows)
            if curve is not None:
                reported.append(TrackedLane(track_id=track.track_id, curve=curve, hits=track.hits, missed=track.missed))
        if config.consistency_rules:
            reported = self._without_crossing_held_lanes(reported, height)
        reported.sort(key=lambda lane: lane.curve.x_at(height - 1))

        ids = {lane.track_id for lane in reported}
        events = TrackingEvents(appeared=sorted(ids - self._reported), disappeared=sorted(self._reported - ids))
        self._reported = ids
        return reported, events

    def note_ego(self, tracked: list[TrackedLane], ego: EgoLane, events: TrackingEvents) -> tuple[int | None, int | None]:
        """Record which tracks bound the ego lane and detect a move into the neighbouring lane.

        The ego lane is always re-selected from the current geometry, so after a lane change no history of the old ego
        lane is carried into the new one: only the lane identities are compared."""
        by_curve = {id(lane.curve): lane.track_id for lane in tracked}
        left_id = by_curve.get(id(ego.left)) if ego.left is not None else None
        right_id = by_curve.get(id(ego.right)) if ego.right is not None else None
        previous_left, previous_right = self._ego_ids
        if left_id is not None and left_id == previous_right:
            events.ego_lane_change = "right"  # the old right boundary is now on the left: the vehicle moved right
        elif right_id is not None and right_id == previous_left:
            events.ego_lane_change = "left"
        if left_id is not None or right_id is not None:
            self._ego_ids = (left_id, right_id)
        return left_id, right_id

    def _distance(self, track: _Track, observed: np.ndarray) -> float | None:
        both = np.isfinite(track.x) & np.isfinite(observed)
        if both.sum() >= 2:
            return float(np.mean(np.abs(track.x[both] - observed[both])))
        # Little overlap (a far dash against a near one): compare with the track's fitted curve, extended.
        known = np.isfinite(track.x)
        seen = np.isfinite(observed)
        if known.sum() < 2 or seen.sum() < 2:
            return None
        rows = self._rows
        coefficients = np.polyfit(rows[known], track.x[known], 1)
        return float(np.mean(np.abs(np.polyval(coefficients, rows[seen]) - observed[seen])))

    def _absorb(self, track: _Track, observed: np.ndarray, probability: float) -> None:
        config = self.config
        seen = np.isfinite(observed)
        known = np.isfinite(track.x)
        both = seen & known
        # The lane's lateral move this frame, as applied to the rows seen in both: unobserved rows follow it.
        shift = config.smoothing * float(np.mean(observed[both] - track.x[both])) if both.any() else 0.0
        updated = track.x.copy()
        updated[both] = config.smoothing * observed[both] + (1.0 - config.smoothing) * track.x[both]
        new_rows = seen & ~known
        updated[new_rows] = observed[new_rows]
        unobserved = known & ~seen
        updated[unobserved] = track.x[unobserved] + shift  # keep a dash gap in place, following the lane's lateral move
        age = track.row_age + 1
        age[seen] = 0
        expired = unobserved & (age > config.row_memory_frames)
        updated[expired] = np.nan
        age[~np.isfinite(updated)] = 0
        track.x = updated
        track.row_age = age
        track.probability = config.smoothing * probability + (1.0 - config.smoothing) * track.probability
        track.hits += 1
        track.missed = 0

    def _forget_inconsistent_rows(self, track: _Track, detection: LaneCurve, gate: float) -> None:
        """Remembered rows (not observed this frame) must continue the detected lane: rows farther than the association
        gate from its extension belonged to something else (another lane, an occluder) and are dropped."""
        remembered = np.isfinite(track.x) & (track.row_age > 0)
        if not remembered.any():
            return
        expected = np.array([detection.x_at(y) for y in self._rows[remembered]])
        stray = np.abs(track.x[remembered] - expected) > gate
        indices = np.flatnonzero(remembered)[stray]
        track.x[indices] = np.nan
        track.row_age[indices] = 0

    def _without_crossing_held_lanes(self, reported: list[TrackedLane], height: int) -> list[TrackedLane]:
        """Lanes do not cross near the vehicle: a held lane that crosses an observed one below mid-height is stale."""
        observed = [lane for lane in reported if not lane.held]
        kept = []
        for lane in reported:
            if lane.held and any(_cross(lane.curve, other.curve, 0.5 * height) for other in observed):
                continue
            kept.append(lane)
        return kept

    def _hold(self, track: _Track) -> None:
        """A frame without a matching detection: the lane keeps its last smoothed geometry."""

    def _curve(self, track: _Track, rows: np.ndarray) -> LaneCurve | None:
        known = np.isfinite(track.x)
        if known.sum() < self.config.min_rows:
            return None
        ys, xs = rows[known], track.x[known]
        degree = self.config.polynomial_degree if len(xs) >= self.config.polynomial_degree + 4 else 1
        coefficients = np.polyfit(ys, xs, degree)
        return LaneCurve(coefficients=coefficients, points=np.stack([xs, ys], axis=1), mean_probability=float(track.probability))


def _cross(a: LaneCurve, b: LaneCurve, from_y: float) -> bool:
    """Whether two lanes swap sides within the rows both were observed on, at or below from_y."""
    top, bottom = max(a.y_top, b.y_top, from_y), min(a.y_bottom, b.y_bottom)
    if bottom - top < 1.0:
        return False
    ys = np.linspace(top, bottom, 12)
    difference = np.array([a.x_at(y) - b.x_at(y) for y in ys])
    return bool(difference.min() < 0.0 < difference.max())


@dataclass(slots=True)
class LaneDepartureConfig:
    """Experimental. Thresholds are fractions of the ego-lane width (|offset| = 0.5 puts the camera on a boundary)."""

    enter_threshold: float = 0.35  # warn when |offset| / lane width stays at or above this ...
    exit_threshold: float = 0.30  # ... and stop once it falls below this (hysteresis)
    min_duration_seconds: float = 0.3  # how long the offset must stay beyond enter_threshold before warning


@dataclass(slots=True)
class DepartureState:
    status: str  # "ok", "warning" or "unavailable" (no reliable ego lane: no judgement is made)
    side: str | None  # "left" or "right" while warning


class LaneDepartureMonitor:
    """Experimental lane departure warning from the normalized vehicle offset (positive: vehicle right of centre).

    No warning is raised from an unreliable ego lane: the state is "unavailable" and the timer restarts. A change of
    ego lane (a completed lane change) also restarts it."""

    def __init__(self, fps: float, config: LaneDepartureConfig | None = None) -> None:
        self.config = config or LaneDepartureConfig()
        self.min_frames = max(1, int(round(self.config.min_duration_seconds * max(fps, 1e-6))))
        self.reset()

    def reset(self) -> None:
        self._side: str | None = None
        self._frames_beyond = 0
        self._warning = False
        self._ego_ids = None
        self.events = 0

    def update(self, offset_normalized: float | None, reliable: bool, ego_ids=None) -> DepartureState:
        if not reliable or offset_normalized is None:
            self._side, self._frames_beyond, self._warning = None, 0, False
            return DepartureState(status="unavailable", side=None)
        if ego_ids is not None and self._ego_ids is not None and ego_ids != self._ego_ids:
            self._side, self._frames_beyond, self._warning = None, 0, False
        if ego_ids is not None:
            self._ego_ids = ego_ids
        side = "right" if offset_normalized > 0 else "left"
        magnitude = abs(offset_normalized)
        if self._warning:
            if magnitude < self.config.exit_threshold or side != self._side:
                self._warning, self._frames_beyond, self._side = False, 0, None
        if not self._warning:
            if magnitude >= self.config.enter_threshold:
                self._frames_beyond = self._frames_beyond + 1 if side == self._side else 1
                self._side = side
                if self._frames_beyond >= self.min_frames:
                    self._warning = True
                    self.events += 1
            else:
                self._side, self._frames_beyond = None, 0
        return DepartureState(status="warning" if self._warning else "ok", side=self._side if self._warning else None)
