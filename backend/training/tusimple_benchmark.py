"""TuSimple lane-detection benchmark for a LaneVision segmentation checkpoint.

The scoring functions reproduce the official TuSimple evaluation (``LaneEval`` in the TuSimple benchmark repository's
evaluate/lane.py): a predicted point matches when it lies within 20 px of the ground truth, widened by 1/cos of the
ground-truth lane angle; a lane counts as matched when at least 85% of its points match; images whose run time
exceeds 200 ms, or with more than len(ground truth) + 2 predicted lanes, score accuracy 0 and FN 1. Accuracy, FP and
FN are averaged over the evaluated images. The angle is the least-squares slope of x over y, as in the original's
scikit-learn LinearRegression.

Predictions come from the production inference path: AIInferenceEngine for the probability map and
app.ai.lane_geometry for multi-lane extraction, sampled at each record's h_samples. Pixel Dice/IoU/precision/recall are
computed alongside at the model input resolution, the same way training scores them.

Usage (from the repository root):
  python -m backend.training.tusimple_benchmark run --checkpoint <ckpt.pth> --labels <test_label.json> \
      --source-root <dataset root> --out-dir <dir> [--only-names <prepared split images dir>] [--device cpu]
  python -m backend.training.tusimple_benchmark summarize --per-image <per_image.jsonl> [--only-names <dir>] --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIXEL_THRESHOLD = 20.0
POINT_THRESHOLD = 0.85
MAX_RUN_TIME_MS = 200.0


def lane_angle(xs, y_samples) -> float:
    """Angle of a ground-truth lane: arctan of the least-squares slope of x over y for its valid points."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(y_samples, dtype=np.float64)
    valid = xs >= 0
    if valid.sum() > 1:
        return float(np.arctan(np.polyfit(ys[valid], xs[valid], 1)[0]))
    return 0.0


def line_accuracy(pred, gt, threshold: float) -> float:
    """Fraction of h_samples where prediction and ground truth agree; rows where both have no lane also agree."""
    pred = np.array([p if p >= 0 else -100 for p in pred], dtype=np.float64)
    gt = np.array([g if g >= 0 else -100 for g in gt], dtype=np.float64)
    return float(np.sum(np.abs(pred - gt) < threshold) / len(gt))


def bench(pred: list[list[float]], gt: list[list[float]], y_samples: list[float], running_time: float) -> tuple[float, float, float]:
    """Accuracy, false-positive rate and false-negative rate of one image, as in the official LaneEval.bench."""
    if any(len(p) != len(y_samples) for p in pred):
        raise ValueError("Format of lanes error.")
    if running_time > MAX_RUN_TIME_MS or len(gt) + 2 < len(pred):
        return 0.0, 0.0, 1.0
    thresholds = [PIXEL_THRESHOLD / np.cos(lane_angle(x_gts, y_samples)) for x_gts in gt]
    line_accs = []
    fn = 0.0
    matched = 0.0
    for x_gts, threshold in zip(gt, thresholds):
        accs = [line_accuracy(x_preds, x_gts, threshold) for x_preds in pred]
        max_acc = max(accs) if accs else 0.0
        if max_acc < POINT_THRESHOLD:
            fn += 1
        else:
            matched += 1
        line_accs.append(max_acc)
    fp = len(pred) - matched
    if len(gt) > 4 and fn > 0:
        fn -= 1
    total = sum(line_accs)
    if len(gt) > 4:
        total -= min(line_accs)
    return (
        total / max(min(4.0, len(gt)), 1.0),
        fp / len(pred) if len(pred) > 0 else 0.0,
        fn / max(min(len(gt), 4.0), 1.0),
    )


def lanes_to_tusimple(lanes, h_samples, width: int, extend_bottom: bool = False) -> list[list[float]]:
    """Sample each extracted lane at the h_samples it was observed across; -2 elsewhere and outside the image.

    extend_bottom: also sample below a lane's lowest observed row, continuing along its end tangent until it leaves the
    image. TuSimple annotations run each lane down to the image bottom, including through occlusions.
    """
    rows = []
    for lane in lanes:
        xs = []
        for y in h_samples:
            inside = lane.y_top <= y <= lane.y_bottom or (extend_bottom and y > lane.y_bottom)
            x = lane.x_at(y) if inside else -2
            xs.append(round(float(x), 1) if 0 <= x < width else -2)
        if any(x >= 0 for x in xs):
            rows.append(xs)
    return rows


def _pixel_counts(predicted: np.ndarray, truth: np.ndarray) -> tuple[int, int, int]:
    return int((predicted & truth).sum()), int((predicted & ~truth).sum()), int((~predicted & truth).sum())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image results: TuSimple means over images, pixel metrics from summed counts."""
    count = len(rows)
    if count == 0:
        return {"images": 0}
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp_px", "fp_px", "fn_px"))
    run_times = np.array([row["run_time_ms"] for row in rows])
    return {
        "images": count,
        "tusimple": {
            "accuracy": sum(row["accuracy"] for row in rows) / count,
            "fp": sum(row["fp"] for row in rows) / count,
            "fn": sum(row["fn"] for row in rows) / count,
            "images_over_run_time_limit": int((run_times > MAX_RUN_TIME_MS).sum()),
        },
        "pixel": {
            "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        },
        "lanes": {
            "ground_truth_mean": sum(row["gt_lanes"] for row in rows) / count,
            "predicted_mean": sum(row["pred_lanes"] for row in rows) / count,
        },
        "run_time_ms": {"mean": float(run_times.mean()), "median": float(np.median(run_times)), "p95": float(np.percentile(run_times, 95))},
    }


def _sample_name(raw_file: str) -> str:
    from backend.training.datasets.tusimple import _sample_key

    return str(Path(_sample_key(raw_file)).with_suffix("")).replace("/", "_").replace("\\", "_").replace(".", "_") + ".png"


def _draw_example(image: np.ndarray, gt_lanes, pred_lanes, h_samples, title: str) -> np.ndarray:
    import cv2

    canvas = image.copy()
    for lane in pred_lanes:
        points = [(int(round(x)), int(y)) for x, y in zip(lane, h_samples) if x >= 0]
        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 3, cv2.LINE_AA)
    for lane in gt_lanes:
        for x, y in zip(lane, h_samples):
            if x >= 0:
                cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import torch

    from backend.app.ai.config import AIConfig
    from backend.app.ai.inference import AIInferenceEngine
    from backend.app.ai.lane_geometry import LaneGeometryConfig, extract_lanes
    from backend.training.datasets.tusimple import _convert_annotation_to_mask, _load_annotation_records, _resolve_image_path

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)
    only_names = {path.name for path in Path(args.only_names).glob("*.png")} if args.only_names else None
    geometry = LaneGeometryConfig(**json.loads(args.geometry)) if args.geometry else LaneGeometryConfig()
    config = AIConfig(enabled=True, model_path=str(checkpoint), device=args.device, mask_threshold=geometry.probability_threshold)
    engine = AIInferenceEngine(config)
    metadata = engine.loader.load()
    engine.run(np.zeros((720, 1280, 3), dtype=np.uint8))  # warm-up, so model loading is not timed as inference
    source_root = Path(args.source_root)
    child_cache: dict = {}

    records = []
    for label_path in args.labels:
        for record in _load_annotation_records(Path(label_path)):
            if only_names is None or _sample_name(record["raw_file"]) in only_names:
                records.append((record, Path(label_path)))
    if args.limit:
        records = records[: args.limit]

    rows = []
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions_file, (out_dir / "per_image.jsonl").open("w", encoding="utf-8") as per_image_file:
        for index, (record, label_path) in enumerate(records):
            image_path = _resolve_image_path(source_root, record["raw_file"], annotation_path=label_path, cache=child_cache)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Unable to read {image_path} for {record['raw_file']}")
            height, width = image.shape[:2]
            h_samples = record["h_samples"]
            started = time.perf_counter()
            inference = engine.run(image)
            probability = cv2.resize(inference["probability_mask"], (width, height), interpolation=cv2.INTER_LINEAR)
            lanes = extract_lanes(probability, geometry, rows=h_samples)
            pred_lanes = lanes_to_tusimple(lanes, h_samples, width, extend_bottom=args.extend_bottom)
            run_time_ms = (time.perf_counter() - started) * 1000.0
            accuracy, fp, fn = bench(pred_lanes, record["lanes"], h_samples, run_time_ms)

            truth = _convert_annotation_to_mask((height, width), record["lanes"], h_samples, lane_thickness=2)
            truth_small = cv2.resize(truth.astype(np.float32), (config.input_width, config.input_height), interpolation=cv2.INTER_NEAREST) >= 0.5
            tp_px, fp_px, fn_px = _pixel_counts(inference["mask"].astype(bool), truth_small)
            row = {"raw_file": record["raw_file"], "sample_name": _sample_name(record["raw_file"]), "label_file": label_path.name,
                   "accuracy": accuracy, "fp": fp, "fn": fn, "gt_lanes": len(record["lanes"]), "pred_lanes": len(pred_lanes),
                   "run_time_ms": run_time_ms, "tp_px": tp_px, "fp_px": fp_px, "fn_px": fn_px}
            rows.append(row)
            predictions_file.write(json.dumps({"raw_file": record["raw_file"], "lanes": pred_lanes, "h_samples": h_samples, "run_time": run_time_ms}) + "\n")
            per_image_file.write(json.dumps(row) + "\n")
            if args.progress and (index + 1) % args.progress == 0:
                print(f"{index + 1}/{len(records)} images, running accuracy {sum(r['accuracy'] for r in rows) / len(rows):.4f}", flush=True)

    # Visual examples: worst, median and best images by TuSimple accuracy.
    ranked = sorted(range(len(rows)), key=lambda i: rows[i]["accuracy"])
    picks = {"worst": ranked[: args.examples], "median": ranked[len(ranked) // 2: len(ranked) // 2 + args.examples], "best": ranked[-args.examples:]} if rows else {}
    examples = []
    predictions = {json.loads(line)["raw_file"]: json.loads(line) for line in (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()}
    for group, indices in picks.items():
        for i in indices:
            record, label_path = records[i]
            image = cv2.imread(str(_resolve_image_path(source_root, record["raw_file"], annotation_path=label_path, cache=child_cache)))
            title = f"{group}: accuracy {rows[i]['accuracy']:.3f}  gt {rows[i]['gt_lanes']} (red)  pred {rows[i]['pred_lanes']} (green)"
            target = out_dir / "examples" / f"{group}_{rows[i]['sample_name'][:-4]}.jpg"
            cv2.imwrite(str(target), _draw_example(image, record["lanes"], predictions[record["raw_file"]]["lanes"], record["h_samples"], title))
            examples.append(str(target.name))

    summary = summarize(rows)
    summary.update({
        "label_files": [str(Path(path)) for path in args.labels],
        "subset": f"names in {args.only_names}" if args.only_names else "all records of the label files",
        "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "epoch": torch.load(str(checkpoint), map_location="cpu").get("epoch")},
        "device": metadata.device,
        "geometry": {name: getattr(geometry, name) for name in geometry.__slots__},
        "extend_bottom": bool(args.extend_bottom),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
        "examples": examples,
    })
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(args.per_image).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only_names:
        names = {path.name for path in Path(args.only_names).glob("*.png")}
        rows = [row for row in rows if row["sample_name"] in names]
    summary = summarize(rows)
    summary["subset"] = f"names in {args.only_names}" if args.only_names else "all rows"
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TuSimple benchmark for LaneVision segmentation checkpoints")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run the checkpoint on TuSimple records and score them")
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--labels", nargs="+", required=True, help="TuSimple label JSON files (e.g. test_label.json)")
    run_parser.add_argument("--source-root", required=True, help="Dataset root used to resolve raw_file paths")
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--only-names", default=None, help="Evaluate only records whose prepared sample name is in this images folder")
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--geometry", default=None, help="JSON object of LaneGeometryConfig overrides")
    run_parser.add_argument("--extend-bottom", action="store_true", help="Extend each predicted lane to the image bottom along its end tangent")
    run_parser.add_argument("--examples", type=int, default=4)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--progress", type=int, default=250)
    summarize_parser = commands.add_parser("summarize", help="Aggregate a per_image.jsonl, optionally for a subset")
    summarize_parser.add_argument("--per-image", required=True)
    summarize_parser.add_argument("--only-names", default=None)
    summarize_parser.add_argument("--out", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run(args) if args.command == "run" else summarize_command(args)
    print(json.dumps({key: value for key, value in result.items() if key not in ("geometry", "environment", "examples")}, indent=2))


if __name__ == "__main__":
    main()
