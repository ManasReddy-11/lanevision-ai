"""Measure temporal lane tracking on TuSimple clips (20 consecutive frames at 20 fps; the 20th frame is annotated).

    cache     run a checkpoint on every frame of each clip; store the lane-probability maps (model resolution, float16)
    evaluate  extract lanes once per frame, then run each tracker variant over the cached clips and report:
              - TuSimple accuracy / FP / FN of the lanes each variant reports for the annotated last frame
                (the official rules without the run-time limit; run time is reported separately)
              - ego-lane offset availability, its frame-to-frame change (jitter), availability flips and gaps
              - lanes reported per frame, distinct lane identities per clip, ego lane changes
              - experimental departure warnings at the default thresholds
              - tracker latency per frame

Variants are compared on the validation split (label_data_0531, prepared val names) to choose the tracker settings;
the test split is then measured once with the chosen settings. Lane extraction and the tracker sample the clip's
h_samples rows, as the single-frame benchmark does.

    python -m backend.training.tusimple_temporal_eval cache --checkpoint lane_segmentation.pth \\
        --labels <root>/TUSimple/train_set/label_data_0531.json --source-root <root> --only-names <prepared>/val/images \\
        --out-dir eval/temporal_val_cache
    python -m backend.training.tusimple_temporal_eval evaluate --cache eval/temporal_val_cache --out eval/temporal_val.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.ai.lane_geometry import LaneGeometryConfig, extract_lanes, select_ego_lane
from backend.app.ai.lane_tracking import LaneDepartureMonitor, LaneTracker, LaneTrackerConfig
from backend.training.tusimple_benchmark import _sample_name, bench, lanes_to_tusimple

FPS = 20.0  # TuSimple clips: 20 frames over one second


def _clip_frames(annotated: Path) -> list[Path]:
    frames = sorted((path for path in annotated.parent.glob("*.jpg") if path.stem.isdigit()), key=lambda path: int(path.stem))
    if not frames or frames[-1].name != annotated.name:
        raise FileNotFoundError(f"{annotated} is not the last frame of its clip folder")
    return frames


def cache(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import torch

    from backend.app.ai.config import AIConfig
    from backend.app.ai.inference import AIInferenceEngine
    from backend.training.datasets.tusimple import _load_annotation_records, _resolve_image_path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only_names = {path.name for path in Path(args.only_names).glob("*.png")} if args.only_names else None
    engine = AIInferenceEngine(AIConfig(enabled=True, model_path=str(args.checkpoint), device=args.device))
    engine.loader.load()
    model, device = engine.loader.model, engine.loader._device or torch.device("cpu")
    source_root = Path(args.source_root)
    child_cache: dict = {}
    records = [(record, Path(label_path)) for label_path in args.labels for record in _load_annotation_records(Path(label_path))
               if only_names is None or _sample_name(record["raw_file"]) in only_names]
    if args.limit:
        records = records[: args.limit]
    index = []
    started = time.perf_counter()
    for position, (record, label_path) in enumerate(records):
        frames = _clip_frames(_resolve_image_path(source_root, record["raw_file"], annotation_path=label_path, cache=child_cache))
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in frames]
        height, width = images[-1].shape[:2]
        batch = torch.cat([engine.preprocessor.prepare(image, device)[0] for image in images])
        with torch.inference_mode():
            output = model(batch)
        logits = output["out"] if isinstance(output, dict) and "out" in output else output
        probabilities = torch.sigmoid(logits[:, engine.config.lane_class_id]).cpu().numpy().astype(np.float16)
        name = _sample_name(record["raw_file"]).rsplit(".", 1)[0] + ".npy"
        np.save(out_dir / name, probabilities)
        index.append({"raw_file": record["raw_file"], "lanes": record["lanes"], "h_samples": record["h_samples"], "width": width,
                      "height": height, "frames": len(frames), "probabilities": name})
        if args.progress and (position + 1) % args.progress == 0:
            print(f"{position + 1}/{len(records)} clips, {time.perf_counter() - started:.0f} s", flush=True)
    (out_dir / "index.json").write_text(json.dumps({"checkpoint": Path(args.checkpoint).name, "clips": index}), encoding="utf-8")
    return {"clips": len(index), "seconds": round(time.perf_counter() - started, 1)}


def _detections(cache_dir: Path, geometry: LaneGeometryConfig) -> list[dict[str, Any]]:
    """Lanes extracted from every cached frame (computed once, stored next to the cache)."""
    import cv2

    stored = cache_dir / "detections.pkl"
    if stored.exists():
        return pickle.loads(stored.read_bytes())
    clips = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))["clips"]
    for clip in clips:
        probabilities = np.load(cache_dir / clip["probabilities"]).astype(np.float32)
        clip["detected"] = [extract_lanes(cv2.resize(p, (clip["width"], clip["height"]), interpolation=cv2.INTER_LINEAR), geometry, rows=clip["h_samples"])
                            for p in probabilities]
    stored.write_bytes(pickle.dumps(clips))
    return clips


class KalmanLaneTracker(LaneTracker):
    """Alternative smoother: a constant-velocity Kalman filter per row (state: x and its change per frame).

    Association, confirmation, holding and row memory are those of LaneTracker; a held lane or unobserved row is
    predicted forward with its velocity instead of kept in place."""

    def __init__(self, process_noise: float, measurement_noise: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.q = process_noise
        self.r = measurement_noise

    def _init_state(self, track) -> dict:
        n = len(track.x)
        return {"v": np.zeros(n), "p00": np.full(n, self.r), "p01": np.zeros(n), "p11": np.full(n, 100.0)}

    def _predict(self, track) -> None:
        state = track.state or self._init_state(track)
        q = self.q
        known = np.isfinite(track.x)
        track.x = np.where(known, track.x + state["v"], track.x)
        state["p00"] = state["p00"] + 2 * state["p01"] + state["p11"] + q / 4
        state["p01"] = state["p01"] + state["p11"] + q / 2
        state["p11"] = state["p11"] + q
        track.state = state

    def _hold(self, track) -> None:
        self._predict(track)

    def _absorb(self, track, observed: np.ndarray, probability: float) -> None:
        self._predict(track)
        state = track.state
        seen = np.isfinite(observed)
        known = np.isfinite(track.x)
        both = seen & known
        s = state["p00"] + self.r
        k0, k1 = state["p00"] / s, state["p01"] / s
        innovation = np.where(both, observed - np.nan_to_num(track.x), 0.0)
        x = track.x.copy()
        x[both] = track.x[both] + k0[both] * innovation[both]
        state["v"] = np.where(both, state["v"] + k1 * innovation, state["v"])
        p00, p01, p11 = state["p00"].copy(), state["p01"].copy(), state["p11"].copy()
        state["p00"] = np.where(both, (1 - k0) * p00, p00)
        state["p01"] = np.where(both, (1 - k0) * p01, p01)
        state["p11"] = np.where(both, p11 - k1 * p01, p11)
        new_rows = seen & ~known
        x[new_rows] = observed[new_rows]
        state["v"][new_rows] = float(np.mean(state["v"][both])) if both.any() else 0.0
        state["p00"][new_rows], state["p01"][new_rows], state["p11"][new_rows] = self.r, 0.0, 100.0
        age = track.row_age + 1
        age[seen] = 0
        expired = known & ~seen & (age > self.config.row_memory_frames)
        x[expired] = np.nan
        age[~np.isfinite(x)] = 0
        track.x, track.row_age = x, age
        track.probability = 0.5 * probability + 0.5 * track.probability
        track.hits += 1
        track.missed = 0


def _make_tracker(variant: dict[str, Any], geometry: LaneGeometryConfig, rows) -> LaneTracker | None:
    kind = variant.get("kind", "ema")
    if kind == "none":
        return None
    settings = {key: value for key, value in variant.items() if key in LaneTrackerConfig.__dataclass_fields__}
    if kind == "ema":
        return LaneTracker(LaneTrackerConfig(**settings), geometry, rows=rows)
    if kind == "kalman":
        return KalmanLaneTracker(variant["process_noise"], variant["measurement_noise"], config=LaneTrackerConfig(**settings), geometry=geometry, rows=rows)
    raise ValueError(f"Unknown tracker kind: {kind}")


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def evaluate_variant(clips: list[dict[str, Any]], variant: dict[str, Any], geometry: LaneGeometryConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_clip = []
    jumps: list[float] = []
    latencies: list[float] = []
    departure_frames = departure_events = lane_changes = 0
    for clip in clips:
        width, height, rows = clip["width"], clip["height"], clip["h_samples"]
        tracker = _make_tracker(variant, geometry, rows)
        monitor = LaneDepartureMonitor(FPS)
        offsets: list[float | None] = []
        identities: set[int] = set()
        lanes_reported = []
        for lanes in clip["detected"]:
            started = time.perf_counter()
            if tracker is None:
                curves, ego_ids = lanes, None
                ego = select_ego_lane(curves, width, height, geometry)
            else:
                tracked, events = tracker.update(lanes, width, height)
                curves = [lane.curve for lane in tracked]
                ego = select_ego_lane(curves, width, height, geometry)
                ego_ids = tracker.note_ego(tracked, ego, events)
                identities.update(lane.track_id for lane in tracked)
                lane_changes += events.ego_lane_change is not None
            latencies.append((time.perf_counter() - started) * 1000.0)
            state = monitor.update(ego.offset_normalized, ego.reliable, ego_ids)
            departure_frames += state.status == "warning"
            offsets.append(ego.offset_pixels if ego.reliable else None)
            lanes_reported.append(len(curves))
        departure_events += monitor.events
        predicted = lanes_to_tusimple(curves, rows, width, extend_bottom=True)
        accuracy, fp, fn = bench(predicted, clip["lanes"], rows, 0.0)
        clip_jumps = [abs(b - a) for a, b in zip(offsets, offsets[1:]) if a is not None and b is not None]
        jumps.extend(clip_jumps)
        available = [offset is not None for offset in offsets]
        gaps, run = [0], 0
        for flag in available:
            run = 0 if flag else run + 1
            gaps.append(run)
        per_clip.append({
            "raw_file": clip["raw_file"], "accuracy": accuracy, "fp": fp, "fn": fn,
            "offset_available": sum(available) / len(available),
            "availability_flips": sum(a != b for a, b in zip(available, available[1:])),
            "longest_gap_frames": max(gaps),
            "mean_offset_jump": statistics.fmean(clip_jumps) if clip_jumps else None,
            "lanes_per_frame": statistics.fmean(lanes_reported),
            "distinct_lane_ids": len(identities) if tracker is not None else None,
            "offsets": [None if offset is None else round(offset, 2) for offset in offsets],
        })
    n = len(per_clip)
    summary = {
        "variant": variant,
        "clips": n,
        "frames": sum(len(clip["detected"]) for clip in clips),
        "last_frame_tusimple": {key: statistics.fmean(row[key] for row in per_clip) for key in ("accuracy", "fp", "fn")},
        "offset_available_rate": statistics.fmean(row["offset_available"] for row in per_clip),
        "offset_jump_px": {"mean": statistics.fmean(jumps) if jumps else None, "median": _percentile(jumps, 50), "p95": _percentile(jumps, 95), "pairs": len(jumps)},
        "availability_flips_per_clip": statistics.fmean(row["availability_flips"] for row in per_clip),
        "longest_gap_frames_mean": statistics.fmean(row["longest_gap_frames"] for row in per_clip),
        "lanes_per_frame": statistics.fmean(row["lanes_per_frame"] for row in per_clip),
        "distinct_lane_ids_per_clip": statistics.fmean(row["distinct_lane_ids"] for row in per_clip) if per_clip and per_clip[0]["distinct_lane_ids"] is not None else None,
        "ego_lane_changes": lane_changes if variant.get("kind", "ema") != "none" else None,
        "departure_warning_frames": departure_frames,
        "departure_events": departure_events,
        "tracking_latency_ms": {"mean": statistics.fmean(latencies), "p95": _percentile(latencies, 95)},
    }
    return summary, per_clip


DEFAULT_VARIANTS = [
    {"name": "none", "kind": "none"},
    *({"name": f"ema_a{a}", "kind": "ema", "smoothing": a} for a in (0.3, 0.5, 0.7, 1.0)),
    {"name": "ema_a0.5_hits1", "kind": "ema", "smoothing": 0.5, "min_hits": 1},
    {"name": "ema_a0.5_hits3", "kind": "ema", "smoothing": 0.5, "min_hits": 3},
    {"name": "ema_a0.5_hold0", "kind": "ema", "smoothing": 0.5, "max_missed_frames": 0},
    {"name": "ema_a0.5_hold6", "kind": "ema", "smoothing": 0.5, "max_missed_frames": 6},
    {"name": "ema_a0.5_rowmem0", "kind": "ema", "smoothing": 0.5, "row_memory_frames": 0},
    {"name": "ema_a0.5_rowmem6", "kind": "ema", "smoothing": 0.5, "row_memory_frames": 6},
    *({"name": f"kalman_q{q}_r{r}", "kind": "kalman", "process_noise": q, "measurement_noise": r} for q in (1.0, 4.0) for r in (25.0, 100.0)),
]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    geometry = LaneGeometryConfig(**json.loads(args.geometry)) if args.geometry else LaneGeometryConfig()
    started = time.perf_counter()
    clips = _detections(Path(args.cache), geometry)
    extraction_seconds = time.perf_counter() - started
    variants = json.loads(Path(args.variants).read_text(encoding="utf-8")) if args.variants else DEFAULT_VARIANTS
    results = []
    per_clip_dir = Path(args.out).with_suffix("")
    per_clip_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        summary, per_clip = evaluate_variant(clips, variant, geometry)
        results.append(summary)
        with (per_clip_dir / f"{variant['name']}.jsonl").open("w", encoding="utf-8") as handle:
            for row in per_clip:
                handle.write(json.dumps(row) + "\n")
        t = summary["last_frame_tusimple"]
        print(f"{variant['name']:20s} acc {t['accuracy']:.4f} fp {t['fp']:.4f} fn {t['fn']:.4f} | offset avail {summary['offset_available_rate']:.3f} "
              f"jump mean {summary['offset_jump_px']['mean']:.2f} p95 {summary['offset_jump_px']['p95']:.2f} | flips {summary['availability_flips_per_clip']:.2f} "
              f"| lanes {summary['lanes_per_frame']:.2f} ids {summary['distinct_lane_ids_per_clip'] or 0:.2f} | departures {summary['departure_events']} "
              f"| {summary['tracking_latency_ms']['mean']:.3f} ms", flush=True)
    report = {"cache": str(Path(args.cache).name), "geometry": json.loads(args.geometry) if args.geometry else "default",
              "lane_extraction_seconds": round(extraction_seconds, 1), "variants": results}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cache_parser = commands.add_parser("cache", help="run the model on every frame of each clip")
    cache_parser.add_argument("--checkpoint", required=True)
    cache_parser.add_argument("--labels", required=True, nargs="+")
    cache_parser.add_argument("--source-root", required=True)
    cache_parser.add_argument("--out-dir", required=True)
    cache_parser.add_argument("--only-names", help="folder of prepared images: keep the clips whose annotated frame is among them")
    cache_parser.add_argument("--device", default="cpu")
    cache_parser.add_argument("--limit", type=int, default=0)
    cache_parser.add_argument("--progress", type=int, default=50)
    evaluate_parser = commands.add_parser("evaluate", help="compare tracker variants on cached clips")
    evaluate_parser.add_argument("--cache", required=True)
    evaluate_parser.add_argument("--out", required=True)
    evaluate_parser.add_argument("--variants", help="JSON file with a list of variants (default: the built-in comparison)")
    evaluate_parser.add_argument("--geometry", help="LaneGeometryConfig overrides as JSON")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = cache(args) if args.command == "cache" else evaluate(args)
    if args.command == "cache":
        print(json.dumps(result))


if __name__ == "__main__":
    main()
