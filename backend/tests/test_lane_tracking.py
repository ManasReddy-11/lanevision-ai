from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app.ai.adapter import AILaneDetectorAdapter
from app.ai.config import AIConfig
from app.ai.lane_geometry import LaneCurve, select_ego_lane
from app.ai.lane_tracking import LaneDepartureConfig, LaneDepartureMonitor, LaneTracker, LaneTrackerConfig
from app.ai.models import LaneSegmentationUNet
from app.services.video_processing_service import VideoProcessingService

W, H = 1280, 720
VANISHING = (640.0, 250.0)
ROWS = np.arange(260.0, 720.0, 10.0)


def lane(bottom_x: float, rows=ROWS, noise: float = 0.0, rng=None, probability: float = 0.8) -> LaneCurve | None:
    """A straight lane through the vanishing point, observed on `rows` where it lies inside the image."""
    vx, vy = VANISHING
    xs = vx + (bottom_x - vx) * (rows - vy) / (H - 1 - vy)
    if noise:
        xs = xs + rng.normal(0.0, noise)  # one lateral error per frame and lane, as the detector's position jitter
    inside = (xs >= 0) & (xs < W)
    if inside.sum() < 4:
        return None
    ys = rows[inside]
    return LaneCurve(coefficients=np.polyfit(ys, xs[inside], 1), points=np.stack([xs[inside], ys], axis=1), mean_probability=probability)


def frame(*bottoms, **kwargs) -> list[LaneCurve]:
    return [curve for curve in (lane(b, **kwargs) for b in bottoms) if curve is not None]


def run(tracker: LaneTracker, frames):
    outputs = []
    for lanes in frames:
        outputs.append(tracker.update(lanes, W, H))
    return outputs


def test_lanes_keep_their_identity_while_they_move() -> None:
    tracker = LaneTracker()
    outputs = run(tracker, [frame(-260 + 4 * t, 340 + 4 * t, 940 + 4 * t, 1540 + 4 * t) for t in range(12)])
    tracked, events = outputs[1]
    assert events.appeared == [1, 2, 3, 4]  # reported once seen in min_hits = 2 frames
    ids = [[lane.track_id for lane in tracked] for tracked, _ in outputs[1:]]
    assert all(row == ids[0] for row in ids) and len(ids[0]) == 4
    assert outputs[0][0] == []  # nothing is reported from a single frame


def test_smoothing_reduces_frame_to_frame_jitter() -> None:
    rng = np.random.default_rng(0)
    frames = [frame(340, 940, noise=6.0, rng=rng) for _ in range(60)]
    tracker = LaneTracker()
    raw, tracked = [], []
    for lanes in frames:
        raw.append(select_ego_lane(lanes, W, H).offset_pixels)
        result, _ = tracker.update(lanes, W, H)
        tracked.append(select_ego_lane([item.curve for item in result], W, H).offset_pixels)
    raw_jitter = np.mean(np.abs(np.diff(raw[2:])))
    tracked_jitter = np.mean(np.abs(np.diff(tracked[2:])))
    assert tracked_jitter < 0.7 * raw_jitter
    assert abs(np.mean(tracked[2:]) - np.mean(raw[2:])) < 3.0  # smoothing, not a bias


def test_a_missed_lane_is_held_briefly_with_decaying_confidence_then_dropped() -> None:
    tracker = LaneTracker(LaneTrackerConfig(max_missed_frames=3))
    run(tracker, [frame(340, 940)] * 5)
    confidences = []
    for missed in range(1, 4):
        tracked, events = tracker.update(frame(940), W, H)
        left = next(item for item in tracked if item.curve.x_at(H - 1) < 640)
        assert left.held and left.missed == missed and events.disappeared == []
        confidences.append(left.curve.mean_probability)
    assert confidences[0] > confidences[1] > confidences[2]
    tracked, events = tracker.update(frame(940), W, H)
    assert [item.curve.x_at(H - 1) > 640 for item in tracked] == [True]
    assert events.disappeared == [1]


def test_a_held_ego_boundary_soon_stops_supporting_a_reliable_offset() -> None:
    tracker = LaneTracker(LaneTrackerConfig(max_missed_frames=5))
    run(tracker, [frame(340, 940, probability=0.6)] * 5)
    reliability = []
    for _ in range(5):
        tracked, _ = tracker.update(frame(940, probability=0.6), W, H)
        reliability.append(select_ego_lane([item.curve for item in tracked], W, H).reliable)
    assert reliability[0] is True  # one missed frame: still a usable boundary
    assert reliability[-1] is False  # its confidence decayed below the ego-lane threshold


def test_a_single_spurious_detection_is_never_reported() -> None:
    tracker = LaneTracker()
    outputs = run(tracker, [frame(340, 940)] * 3 + [frame(340, 940, 1100)] + [frame(340, 940)] * 3)
    assert all(len(tracked) == 2 for tracked, _ in outputs[1:])
    assert all(events.appeared == [] for _, events in outputs[2:])


def test_dashed_lane_keeps_its_full_extent_across_dash_gaps() -> None:
    near, far = ROWS[ROWS >= 500], ROWS[ROWS < 500]
    frames = [[lane(340, rows=near if t % 2 else far), lane(940)] for t in range(8)]
    tracked, _ = run(LaneTracker(), frames)[-1]
    left = tracked[0].curve
    assert left.y_top <= 270 and left.y_bottom >= 700  # the union of recently observed dashes
    assert abs(left.x_at(H - 1) - 340) < 3


def test_lane_change_switches_the_ego_lane_without_carrying_old_geometry() -> None:
    tracker = LaneTracker()
    offsets, changes, ego_pairs = [], [], []
    for t in range(36):
        shift = -20.0 * max(0, t - 5)  # the vehicle moves right: the lanes move left in the image
        tracked, events = tracker.update(frame(-260 + shift, 340 + shift, 940 + shift, 1540 + shift), W, H)
        ego = select_ego_lane([item.curve for item in tracked], W, H)
        ego_pairs.append(tracker.note_ego(tracked, ego, events))
        changes.append(events.ego_lane_change)
        offsets.append(ego.offset_pixels)
    assert [change for change in changes if change] == ["right"]
    before, after = ego_pairs[5], ego_pairs[-1]
    assert after[0] == before[1]  # the old right boundary is the new left boundary
    lanes_now = 940 - 20.0 * 30
    expected = 640.0 - (lanes_now + lanes_now + 600) / 2.0
    assert offsets[-1] == pytest.approx(expected, abs=25.0)  # measured in the new lane (EMA lag of a few px only)


def test_reset_and_a_new_frame_size_start_from_scratch() -> None:
    tracker = LaneTracker()
    run(tracker, [frame(340, 940)] * 3)
    tracker.reset()
    tracked, _ = tracker.update(frame(340, 940), W, H)
    assert tracked == []
    tracked, events = tracker.update(frame(340, 940), W, H)
    assert [item.track_id for item in tracked] == [1, 2] and events.appeared == [1, 2]
    smaller, _ = tracker.update(frame(340, 940), 640, 360)
    assert smaller == []  # a different video size cannot continue the previous tracks


def test_departure_warning_needs_a_sustained_offset_and_uses_hysteresis() -> None:
    monitor = LaneDepartureMonitor(fps=20.0, config=LaneDepartureConfig(enter_threshold=0.35, exit_threshold=0.30, min_duration_seconds=0.3))
    states = [monitor.update(0.4, True) for _ in range(6)]
    assert [state.status for state in states] == ["ok"] * 5 + ["warning"] and states[-1].side == "right"
    assert monitor.update(0.32, True).status == "warning"  # between the thresholds: stays on
    assert monitor.update(0.25, True).status == "ok"
    assert monitor.events == 1
    assert [monitor.update(-0.4, True).side for _ in range(6)][-1] == "left"


def test_departure_warning_is_withheld_without_a_reliable_ego_lane() -> None:
    monitor = LaneDepartureMonitor(fps=20.0)
    for _ in range(5):
        monitor.update(0.45, True)
    assert monitor.update(None, False).status == "unavailable"
    assert [monitor.update(0.45, True).status for _ in range(5)] == ["ok"] * 5  # the timer restarted
    monitor.update(0.45, True, ego_ids=(3, 4))
    assert monitor.update(0.45, True, ego_ids=(4, 5)).status == "ok"  # a new ego lane restarts it too


def _probability_video(tmp_path: Path, bottoms_per_frame) -> tuple[Path, list[np.ndarray]]:
    probabilities = []
    for bottoms in bottoms_per_frame:
        probability = np.zeros((160, 320), dtype=np.float32)
        for bottom in bottoms:
            cv2.line(probability, (160, 56), (int(bottom / 4), 159), 0.9, 2)
        probabilities.append(probability)
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
    for _ in probabilities:
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    writer.release()
    return path, probabilities


class _Upload:
    def __init__(self, path: Path) -> None:
        self.filename = path.name
        self.content_type = "video/mp4"
        self.file = open(path, "rb")


def test_video_tracks_lanes_per_video_and_reports_temporal_metrics(tmp_path, monkeypatch) -> None:
    torch.manual_seed(0)
    checkpoint = tmp_path / "unet.pth"
    torch.save({"state_dict": LaneSegmentationUNet(in_channels=3, num_classes=2).state_dict()}, checkpoint)
    adapter = AILaneDetectorAdapter(AIConfig(enabled=True, model_path=str(checkpoint), device="cpu", input_width=320, input_height=160))
    # Ten frames: the left boundary is missed in frame 6 (held), then the frames repeat for a second video.
    bottoms = [(40, 400, 900, 1240)] * 6 + [(40, 900, 1240)] + [(40, 400, 900, 1240)] * 3
    path, probabilities = _probability_video(tmp_path, bottoms)
    stream = iter(probabilities * 2)

    class FixedEngine:
        loader = adapter.engine.loader

        def run(self, frame):
            probability = next(stream)
            return {"probability_mask": probability, "mask": (probability >= 0.5).astype(np.uint8), "model_metadata": None}

    adapter.engine = FixedEngine()
    monkeypatch.setattr("app.services.video_processing_service.get_detector", lambda mode: adapter)
    service = VideoProcessingService()
    first = service.process_upload(_Upload(path), algorithm="ai")
    second = service.process_upload(_Upload(path), algorithm="ai")

    for response in (first, second):
        assert response.metrics.frames_processed == 10
        assert response.metrics.vehicle_offset_available_rate == pytest.approx(0.9)  # frame 1 only: lanes not yet confirmed
        assert response.metrics.vehicle_offset_jitter_pixels is not None and response.metrics.vehicle_offset_jitter_pixels < 5.0
        assert response.metrics.lane_departure_events == 0
        assert response.ai_metadata["tracking"] == {"distinct_lanes": 4, "frames_with_held_ego_boundary": 1, "ego_lane_changes": 0}
        assert "Ego-lane boundary held from previous frames (not detected in this frame) (1/10 frames)" in response.warnings


def test_a_held_lane_that_crosses_an_observed_lane_is_not_reported() -> None:
    tracker = LaneTracker()
    crossing = LaneCurve(coefficients=np.array([-2.0, 1900.0]), points=np.array([[1900 - 2.0 * y, y] for y in ROWS if 0 <= 1900 - 2.0 * y < W]), mean_probability=0.8)
    run(tracker, [frame(340, 940) + [crossing]] * 3)  # a stray line, confirmed, crossing the right boundary near the car
    tracked, _ = tracker.update(frame(340, 940), W, H)  # the stray line is not detected any more: it would be held
    assert len(tracked) == 2 and not any(lane.held for lane in tracked)
    no_rules = LaneTracker(LaneTrackerConfig(consistency_rules=False))
    run(no_rules, [frame(340, 940) + [crossing]] * 3)
    held, _ = no_rules.update(frame(340, 940), W, H)
    assert len(held) == 3  # without the rule the stale, crossing line stays on screen


def test_remembered_rows_that_disagree_with_the_detection_are_dropped() -> None:
    # A detection that matches the lane near the car (within the association gate) but points elsewhere further up:
    # the lane's remembered far rows no longer continue it and must not be kept.
    def near_rows_turning_away() -> LaneCurve:
        base = lane(940)
        ys = ROWS[ROWS >= 600]
        xs = np.array([base.x_at(y) + 0.5 * (y - 660) for y in ys])
        return LaneCurve(coefficients=np.polyfit(ys, xs, 1), points=np.stack([xs, ys], axis=1), mean_probability=0.8)

    for rules in (True, False):
        tracker = LaneTracker(LaneTrackerConfig(consistency_rules=rules))
        run(tracker, [frame(940)] * 3)
        detection = near_rows_turning_away()
        tracked, _ = tracker.update([detection], W, H)
        assert len(tracked) == 1 and not tracked[0].held  # the detection was associated with the lane
        points = tracked[0].curve.points
        deviation = np.abs(points[:, 0] - np.array([detection.x_at(y) for y in points[:, 1]]))
        if rules:
            assert deviation.max() <= 0.04 * W
        else:
            assert deviation.max() > 0.04 * W  # without the rule the old far rows are kept
