from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load_annotation_records(annotation_path: Path) -> list[dict[str, Any]]:
    if not annotation_path.exists():
        raise FileNotFoundError(f"TuSimple annotation file not found: {annotation_path}")

    text = annotation_path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if payload is not None:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload, dict) and isinstance(payload.get("lanes"), list):
            return [payload]
        raise ValueError("Unsupported TuSimple annotation format")

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON line in {annotation_path}: {exc}") from exc
        if isinstance(record, dict):
            records.append(record)
        else:
            raise ValueError(f"Unsupported TuSimple JSON line format in {annotation_path}")
    return records


def build_split_manifest(records: list[str], seed: int = 42, validation_ratio: float = 0.1, test_ratio: float = 0.1) -> dict[str, list[str]]:
    if not 0.0 <= validation_ratio <= 1.0:
        raise ValueError("validation_ratio must be in [0, 1]")
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError("test_ratio must be in [0, 1]")

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    total = len(shuffled)
    validation_count = max(1, int(round(total * validation_ratio))) if total else 0
    test_count = max(1, int(round(total * test_ratio))) if total else 0
    if validation_count + test_count >= total:
        validation_count = max(1, total // 2)
        test_count = max(0, total - validation_count)

    validation_items = shuffled[:validation_count]
    test_items = shuffled[validation_count : validation_count + test_count]
    train_items = shuffled[validation_count + test_count :]
    return {"train": train_items, "val": validation_items, "test": test_items}


def _draw_lane_line(mask: np.ndarray, points: list[tuple[int, int]], thickness: int) -> None:
    if len(points) < 2:
        return
    for index in range(1, len(points)):
        prev = points[index - 1]
        current = points[index]
        cv2.line(mask, prev, current, 255, thickness=thickness, lineType=cv2.LINE_AA)


def _convert_annotation_to_mask(image_shape: tuple[int, int], lanes: list[list[float | int]], h_samples: list[float | int], lane_thickness: int) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not lanes or not h_samples:
        return mask

    has_any_valid_points = False
    for lane_coords in lanes:
        if not isinstance(lane_coords, list):
            continue
        valid_points: list[tuple[int, int]] = []
        for sample_index, sample in enumerate(h_samples):
            value = lane_coords[sample_index] if sample_index < len(lane_coords) else None
            if value is None:
                continue
            try:
                x_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(x_value) or math.isinf(x_value) or x_value < 0:
                continue
            y_value = float(sample)
            if y_value < 0 or y_value >= height:
                continue
            x_pixel = int(round(x_value))
            if x_pixel < 0 or x_pixel >= width:
                continue
            valid_points.append((x_pixel, int(round(y_value))))
        if len(valid_points) >= 2:
            _draw_lane_line(mask, valid_points, lane_thickness)
            has_any_valid_points = True
    return mask if has_any_valid_points else np.zeros((height, width), dtype=np.uint8)


def _resolve_image_path(source_root: Path, raw_file: str, annotation_path: Path | None = None) -> Path:
    candidate = Path(raw_file)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        search_roots: list[Path] = [source_root]
        if annotation_path is not None:
            annotation_dir = annotation_path.parent
            search_roots.extend([
                annotation_dir,
                annotation_dir / "clips",
                annotation_dir.parent,
                annotation_dir.parent / "clips",
                source_root / annotation_dir.name,
                source_root / annotation_dir.name / "clips",
            ])

        for child in sorted(source_root.iterdir()):
            if child.is_dir():
                search_roots.extend([
                    child,
                    child / "clips",
                    source_root / child.name,
                    source_root / child.name / "clips",
                ])

        search_roots.extend([
            source_root / "clips",
            source_root / candidate.name,
            source_root / "clips" / candidate.name,
        ])

        for root in search_roots:
            candidates.append(root / candidate)
            candidates.append(root / candidate.name)
            candidates.append(root / "clips" / candidate)
            candidates.append(root / "clips" / candidate.name)

    unique_candidates: list[Path] = []
    for path in candidates:
        if path not in unique_candidates:
            unique_candidates.append(path)

    for path in unique_candidates:
        if path.exists():
            return path

    if candidate.parts and len(candidate.parts) > 1:
        fallback = source_root / candidate.name
        if fallback.exists():
            return fallback
    return unique_candidates[0] if unique_candidates else source_root / candidate


def _discover_annotation_files(source_root: Path) -> list[Path]:
    candidates = []
    for path in sorted(source_root.rglob("*.json")):
        if path.name.startswith("."):
            continue
        if any(token in path.name.lower() for token in ("label", "task", "train_val", "test")):
            candidates.append(path)
    if not candidates:
        for path in sorted(source_root.rglob("*.json")):
            candidates.append(path)
    return candidates


def inspect_tusimple_source(source_root: str | Path, sample_size: int = 8) -> dict[str, Any]:
    source_root = Path(source_root)
    annotation_files = _discover_annotation_files(source_root)
    clip_dirs = sorted({str(path) for path in source_root.rglob("clips") if path.is_dir()})
    image_dirs = sorted({str(path.parent) for path in source_root.rglob("*.jpg")})

    samples: list[dict[str, Any]] = []
    resolved_samples = 0
    sampled_records = 0
    for annotation_path in annotation_files:
        try:
            records = _load_annotation_records(annotation_path)
        except ValueError:
            continue
        for record in records:
            raw_file = record.get("raw_file")
            if not raw_file:
                continue
            resolved_path = _resolve_image_path(source_root, raw_file, annotation_path=annotation_path)
            exists = resolved_path.exists()
            sampled_records += 1
            if exists:
                resolved_samples += 1
            if len(samples) < sample_size:
                samples.append(
                    {
                        "annotation_file": str(annotation_path),
                        "raw_file": raw_file,
                        "resolved_path": str(resolved_path),
                        "exists": exists,
                    }
                )
            if len(samples) >= sample_size and sampled_records >= sample_size:
                break
        if len(samples) >= sample_size and sampled_records >= sample_size:
            break

    return {
        "source_root": str(source_root),
        "annotation_files": [str(path) for path in annotation_files],
        "clip_dirs": clip_dirs,
        "image_dirs": image_dirs[:25],
        "sample_records": samples,
        "sampled_record_count": sampled_records,
        "resolved_record_count": resolved_samples,
    }


def discover_tusimple_source_root(search_root: str | Path, sample_size: int = 8) -> dict[str, Any]:
    search_root = Path(search_root)
    candidate_roots = [search_root]
    if search_root.exists():
        candidate_roots.extend(sorted(path for path in search_root.iterdir() if path.is_dir()))

    inspections: list[dict[str, Any]] = []
    best_match: dict[str, Any] | None = None
    best_score = (-1, -1)
    seen: set[Path] = set()
    for candidate_root in candidate_roots:
        if candidate_root in seen or not candidate_root.exists():
            continue
        seen.add(candidate_root)
        inspection = inspect_tusimple_source(candidate_root, sample_size=sample_size)
        annotation_count = len(inspection["annotation_files"])
        resolved_count = int(inspection["resolved_record_count"])
        has_split_dirs = any((candidate_root / name).is_dir() for name in ("train_set", "test_set"))
        direct_annotation_count = len(list(candidate_root.glob("*.json")))
        inspection["score"] = {
            "annotation_count": annotation_count,
            "resolved_record_count": resolved_count,
            "has_split_dirs": has_split_dirs,
            "direct_annotation_count": direct_annotation_count,
        }
        inspections.append(inspection)
        score = (resolved_count, int(has_split_dirs), direct_annotation_count, annotation_count)
        if score > best_score:
            best_score = score
            best_match = inspection

    if best_match is None or best_score[0] <= 0:
        raise FileNotFoundError(f"Unable to discover a usable TuSimple source root under {search_root}")

    return {
        "search_root": str(search_root),
        "best_match": best_match,
        "candidates_considered": inspections,
    }


def _normalize_raw_file(raw_file: str) -> str:
    normalized = str(raw_file).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _ensure_empty_output_root(output_root: Path) -> None:
    for split_name in ("train", "val", "test"):
        for folder_name in ("images", "masks"):
            folder = output_root / split_name / folder_name
            if folder.is_dir() and any(folder.iterdir()):
                raise FileExistsError(
                    f"Output folder {folder} already contains files. Prepare into an empty output root so that "
                    "stale samples from an earlier run cannot leak across splits."
                )


def convert_tusimple_to_dataset(source_root: str | Path, output_root: str | Path, lane_thickness: int = 2, split_seed: int = 42, validation_ratio: float = 0.1, test_ratio: float = 0.1) -> dict[str, Any]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    annotation_files = _discover_annotation_files(source_root)
    if not annotation_files:
        raise FileNotFoundError(f"No TuSimple annotation files found under {source_root}")
    _ensure_empty_output_root(output_root)

    record_entries: list[tuple[dict[str, Any], Path]] = []
    for annotation_path in annotation_files:
        if annotation_path.name.lower() in {"test.json", "train_val.json"}:
            continue
        if "seg_label" in str(annotation_path):
            continue
        try:
            for record in _load_annotation_records(annotation_path):
                record_entries.append((record, annotation_path))
        except ValueError as exc:
            raise ValueError(f"Unable to parse {annotation_path}: {exc}") from exc

    manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "annotation_files": [str(path) for path in annotation_files],
        "total_annotations": len(record_entries),
        "valid_pairs": 0,
        "invalid_missing_images": 0,
        "invalid_annotations": 0,
        "masks_generated": 0,
        "empty_masks": 0,
        "duplicate_references": 0,
        "conflicting_duplicates": 0,
        "ambiguous_samples": 0,
        "split": {},
    }

    for split_name in ("train", "val", "test"):
        for folder_name in ("images", "masks"):
            (output_root / split_name / folder_name).mkdir(parents=True, exist_ok=True)

    # Each image may become exactly one sample. Records are validated first; among valid records, annotation
    # files are read in sorted order, so the first valid reference to a raw_file wins deterministically and
    # later valid references are counted as duplicates instead of being converted again.
    converted_raw_files: dict[str, dict[str, Any]] = {}
    claimed_images: dict[str, str] = {}
    claimed_output_names: dict[str, str] = {}
    sample_names = []
    for record, annotation_path in record_entries:
        raw_file = record.get("raw_file")
        lanes = record.get("lanes", [])
        h_samples = record.get("h_samples", [])
        if not isinstance(lanes, list) or not isinstance(h_samples, list) or not raw_file:
            manifest["invalid_annotations"] += 1
            continue

        image_path = _resolve_image_path(source_root, raw_file, annotation_path=annotation_path)
        if not image_path.exists():
            manifest["invalid_missing_images"] += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            manifest["invalid_missing_images"] += 1
            continue

        mask = _convert_annotation_to_mask(image.shape[:2], lanes, h_samples, lane_thickness)
        if mask.sum() == 0:
            manifest["invalid_annotations"] += 1
            manifest["empty_masks"] += 1
            continue

        raw_key = _normalize_raw_file(raw_file)
        if raw_key in converted_raw_files:
            manifest["duplicate_references"] += 1
            previous = converted_raw_files[raw_key]
            if previous.get("lanes") != lanes or previous.get("h_samples") != h_samples:
                manifest["conflicting_duplicates"] += 1
            continue

        # Two different raw_file values must never share one image or one output name.
        image_key = os.path.normcase(os.path.abspath(str(image_path)))
        safe_stem = str(Path(raw_key).with_suffix(""))
        safe_stem = safe_stem.replace("/", "_").replace("\\", "_").replace(".", "_")
        if image_key in claimed_images or safe_stem in claimed_output_names:
            manifest["ambiguous_samples"] += 1
            continue

        manifest["masks_generated"] += 1

        converted_raw_files[raw_key] = record
        claimed_images[image_key] = raw_key
        claimed_output_names[safe_stem] = raw_key
        image_output_path = output_root / "train" / "images" / f"{safe_stem}.png"
        mask_output_path = output_root / "train" / "masks" / f"{safe_stem}.png"
        cv2.imwrite(str(image_output_path), image)
        cv2.imwrite(str(mask_output_path), mask)
        sample_names.append(f"{safe_stem}.png")
        manifest["valid_pairs"] += 1

    split_manifest = build_split_manifest(sample_names, seed=split_seed, validation_ratio=validation_ratio, test_ratio=test_ratio)
    manifest["split"] = split_manifest

    # Every sample was staged in train/. Move (not copy) val and test samples out so the splits stay disjoint.
    for split_name in ("val", "test"):
        split_dir = output_root / split_name
        for item in split_manifest.get(split_name, []):
            for folder_name in ("images", "masks"):
                staged = output_root / "train" / folder_name / item
                if staged.exists():
                    staged.replace(split_dir / folder_name / item)

    return manifest


def generate_preview_samples(dataset_root: str | Path, output_dir: str | Path | None = None, limit: int = 4) -> list[Path]:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir or dataset_root / ".." / "training_outputs" / "previews")
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_paths: list[Path] = []
    for split_name in ("train", "val", "test"):
        images_dir = dataset_root / split_name / "images"
        masks_dir = dataset_root / split_name / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            continue
        image_paths = sorted(images_dir.glob("*.png"))
        for image_path in image_paths[:limit]:
            stem = image_path.stem
            mask_path = masks_dir / f"{stem}.png"
            if not mask_path.exists():
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                continue
            overlay = image.copy()
            overlay[mask > 0] = [0, 255, 0]
            preview_path = output_dir / f"{split_name}_{stem}_preview.png"
            cv2.imwrite(str(preview_path), np.hstack([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay]))
            preview_paths.append(preview_path)
            if len(preview_paths) >= limit:
                return preview_paths
    return preview_paths


def validate_dataset(dataset_root: str | Path) -> dict[str, int]:
    dataset_root = Path(dataset_root)
    counts = {
        "train_count": 0,
        "val_count": 0,
        "validation_count": 0,
        "test_count": 0,
        "unpaired_files": 0,
        "train_val_overlap": 0,
        "train_test_overlap": 0,
        "val_test_overlap": 0,
    }
    names: dict[str, set[str]] = {}
    for split_name in ("train", "val", "test"):
        images_dir = dataset_root / split_name / "images"
        masks_dir = dataset_root / split_name / "masks"
        image_names = {path.name for path in images_dir.glob("*.png")} if images_dir.exists() else set()
        mask_names = {path.name for path in masks_dir.glob("*.png")} if masks_dir.exists() else set()
        if images_dir.exists() and masks_dir.exists():
            counts[f"{split_name}_count"] = len(image_names)
        counts["unpaired_files"] += len(image_names ^ mask_names)
        names[split_name] = image_names
    counts["validation_count"] = counts.get("val_count", counts["validation_count"])
    counts["train_val_overlap"] = len(names["train"] & names["val"])
    counts["train_test_overlap"] = len(names["train"] & names["test"])
    counts["val_test_overlap"] = len(names["val"] & names["test"])
    return counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TuSimple dataset preparation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser("discover", help="Inspect a TuSimple source root or search tree")
    discover_parser.add_argument("--dataset", required=True)
    discover_parser.add_argument("--sample-size", type=int, default=8)
    validate_parser = subparsers.add_parser("validate", help="Validate or prepare a TuSimple dataset")
    validate_parser.add_argument("--dataset", required=True)
    validate_parser.add_argument("--lane-thickness", type=int, default=2)
    validate_parser.add_argument("--split-seed", type=int, default=42)
    preview_parser = subparsers.add_parser("preview", help="Generate preview images for converted masks")
    preview_parser.add_argument("--dataset", required=True)
    preview_parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "discover":
        report = discover_tusimple_source_root(args.dataset, sample_size=args.sample_size)
        print(json.dumps(report, indent=2))
    elif args.command == "validate":
        manifest = convert_tusimple_to_dataset(args.dataset, args.dataset + "_prepared", lane_thickness=args.lane_thickness, split_seed=args.split_seed)
        print(json.dumps(manifest, indent=2))
    elif args.command == "preview":
        preview_paths = generate_preview_samples(args.dataset, output_dir=args.output_dir)
        print(json.dumps([str(path) for path in preview_paths], indent=2))


if __name__ == "__main__":
    main()
