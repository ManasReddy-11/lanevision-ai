from __future__ import annotations

import numpy as np

from backend.app.ai.lane_geometry import LaneCurve, LaneGeometryConfig
from backend.training.tusimple_temporal_eval import _cached, evaluate_variant

ROWS = list(range(240, 720, 10))


def _lane(bottom_x: float, jitter: float = 0.0) -> LaneCurve:
    ys = np.array(ROWS, dtype=float)
    xs = 640.0 + (bottom_x - 640.0) * (ys - 230.0) / (719.0 - 230.0) + jitter
    return LaneCurve(coefficients=np.polyfit(ys, xs, 1), points=np.stack([xs, ys], axis=1), mean_probability=0.8)


def test_cached_clips_are_reused_only_when_complete(tmp_path) -> None:
    np.save(tmp_path / "complete.npy", np.zeros((20, 4, 8), dtype=np.float16))
    np.save(tmp_path / "short.npy", np.zeros((12, 4, 8), dtype=np.float16))
    (tmp_path / "broken.npy").write_bytes(b"not an array")
    assert _cached(tmp_path / "complete.npy", 20)
    assert not _cached(tmp_path / "short.npy", 20)
    assert not _cached(tmp_path / "broken.npy", 20)
    assert not _cached(tmp_path / "missing.npy", 20)


def test_variants_are_scored_on_the_last_frame_and_for_stability() -> None:
    rng = np.random.default_rng(0)
    truth = [[float(x) for x in _lane(b).sample(ROWS)] for b in (340, 940)]
    detected = [[_lane(340, rng.normal(0, 6)), _lane(940, rng.normal(0, 6))] for _ in range(19)] + [[_lane(940)]]  # left lane missed last
    clip = {"raw_file": "clips/x/20.jpg", "width": 1280, "height": 720, "h_samples": ROWS, "lanes": truth, "detected": detected}
    geometry = LaneGeometryConfig()
    raw, _ = evaluate_variant([clip], {"name": "none", "kind": "none"}, geometry)
    tracked, rows = evaluate_variant([clip], {"name": "ema", "kind": "ema"}, geometry)
    assert raw["last_frame_tusimple"]["fn"] == 0.5  # the single-frame result misses the left lane on frame 20
    assert tracked["last_frame_tusimple"]["accuracy"] == 1.0  # the tracker holds it
    assert tracked["offset_jump_px"]["mean"] < raw["offset_jump_px"]["mean"]
    assert rows[0]["distinct_lane_ids"] == 2
