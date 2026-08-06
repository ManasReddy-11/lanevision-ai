from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.training.datasets.tusimple import _discover_annotation_files, _load_annotation_records


def _sanitize_output_stem(raw_file: str) -> str:
    return str(Path(raw_file).with_suffix("")) .replace("/", "_").replace("\\", "_").replace(".", "_")


def _collect_split_files(dataset_root: Path, split: str) -> list[Path]:
    images_dir = dataset_root / split / "images"
    masks_dir = dataset_root / split / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        return []
    return sorted(images_dir.glob("*.png"))


def _load_mask(mask_path: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _validate_sample(image_path: Path, mask_path: Path, split_name: str, sample_index: int, output_dir: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = _load_mask(mask_path)
    if image is None:
        return {"split": split_name, "sample_index": sample_index, "ok": False, "reason": "unreadable_image"}
    if mask is None:
        return {"split": split_name, "sample_index": sample_index, "ok": False, "reason": "unreadable_mask"}

    if image.shape[:2] != mask.shape[:2]:
        return {"split": split_name, "sample_index": sample_index, "ok": False, "reason": "dimension_mismatch"}

    positive_pixels = int((mask > 0).sum())
    total_pixels = int(mask.size)
    occupied_ratio = positive_pixels / total_pixels if total_pixels else 0.0
    if positive_pixels == 0:
        return {"split": split_name, "sample_index": sample_index, "ok": False, "reason": "empty_mask", "occupied_ratio": occupied_ratio}
    if occupied_ratio > 0.2:
        return {"split": split_name, "sample_index": sample_index, "ok": False, "reason": "excessive_occupancy", "occupied_ratio": occupied_ratio}

    overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]
    preview = np.hstack([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay])
    preview_path = output_dir / f"{split_name}_{sample_index:02d}_{image_path.stem}_preview.png"
    cv2.imwrite(str(preview_path), preview)

    return {
        "split": split_name,
        "sample_index": sample_index,
        "ok": True,
        "reason": "ok",
        "positive_pixels": positive_pixels,
        "occupied_ratio": occupied_ratio,
        "preview_path": str(preview_path),
    }


def _collect_dataset_stats(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "splits": {},
        "total_masks": 0,
        "non_empty_masks": 0,
        "empty_masks": 0,
        "unexpected_mask_values": 0,
        "unreadable_images": 0,
        "unreadable_masks": 0,
        "dimension_mismatches": 0,
        "positive_pixel_counts": [],
        "positive_pixel_percentages": [],
    }

    for split_name in ("train", "val", "test"):
        images_dir = dataset_root / split_name / "images"
        masks_dir = dataset_root / split_name / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            continue
        image_paths = sorted(images_dir.glob("*.png"))
        split_stats = {
            "image_count": len(image_paths),
            "mask_count": 0,
            "non_empty_masks": 0,
            "empty_masks": 0,
            "unexpected_mask_values": 0,
            "unreadable_images": 0,
            "unreadable_masks": 0,
            "dimension_mismatches": 0,
            "positive_pixel_counts": [],
            "positive_pixel_percentages": [],
        }
        for image_path in image_paths:
            mask_path = masks_dir / image_path.name
            if not mask_path.exists():
                split_stats["unreadable_masks"] += 1
                continue
            split_stats["mask_count"] += 1
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = _load_mask(mask_path)
            if image is None:
                split_stats["unreadable_images"] += 1
                continue
            if mask is None:
                split_stats["unreadable_masks"] += 1
                continue
            if image.shape[:2] != mask.shape[:2]:
                split_stats["dimension_mismatches"] += 1
                continue
            positive_pixels = int((mask > 0).sum())
            total_pixels = int(mask.size)
            occupied_ratio = positive_pixels / total_pixels if total_pixels else 0.0
            split_stats["positive_pixel_counts"].append(positive_pixels)
            split_stats["positive_pixel_percentages"].append(occupied_ratio * 100.0)
            if positive_pixels > 0:
                split_stats["non_empty_masks"] += 1
            else:
                split_stats["empty_masks"] += 1

        stats["splits"][split_name] = split_stats
        stats["total_masks"] += split_stats["mask_count"]
        stats["non_empty_masks"] += split_stats["non_empty_masks"]
        stats["empty_masks"] += split_stats["empty_masks"]
        stats["unexpected_mask_values"] += split_stats["unexpected_mask_values"]
        stats["unreadable_images"] += split_stats["unreadable_images"]
        stats["unreadable_masks"] += split_stats["unreadable_masks"]
        stats["dimension_mismatches"] += split_stats["dimension_mismatches"]
        stats["positive_pixel_counts"].extend(split_stats["positive_pixel_counts"])
        stats["positive_pixel_percentages"].extend(split_stats["positive_pixel_percentages"])

    if stats["positive_pixel_counts"]:
        stats["min_positive_pixels"] = min(stats["positive_pixel_counts"])
        stats["max_positive_pixels"] = max(stats["positive_pixel_counts"])
        stats["mean_positive_pixels"] = float(np.mean(stats["positive_pixel_counts"]))
        stats["mean_lane_pixel_percentage"] = float(np.mean(stats["positive_pixel_percentages"]))
    else:
        stats["min_positive_pixels"] = 0
        stats["max_positive_pixels"] = 0
        stats["mean_positive_pixels"] = 0.0
        stats["mean_lane_pixel_percentage"] = 0.0

    return stats


def _sample_annotation_alignment(dataset_root: Path, source_root: Path, output_dir: Path, count: int = 10) -> dict[str, Any]:
    annotation_files = _discover_annotation_files(source_root)
    records: list[dict[str, Any]] = []
    for path in annotation_files:
        try:
            records.extend(_load_annotation_records(path))
        except ValueError:
            continue

    rng = random.Random(7)
    sampled = rng.sample(records, k=min(count, len(records)))

    results: list[dict[str, Any]] = []
    for record in sampled:
        raw_file = record.get("raw_file")
        lanes = record.get("lanes", [])
        h_samples = record.get("h_samples", [])
        if not raw_file or not isinstance(lanes, list) or not isinstance(h_samples, list):
            continue

        output_stem = _sanitize_output_stem(raw_file)
        mask_found = False
        for split_name in ("train", "val", "test"):
            mask_path = dataset_root / split_name / "masks" / f"{output_stem}.png"
            if mask_path.exists():
                mask = _load_mask(mask_path)
                if mask is None:
                    continue
                mask_found = True
                image_shape = mask.shape[:2]
                aligned = False
                for lane_coords in lanes:
                    if not isinstance(lane_coords, list):
                        continue
                    for sample_idx, sample in enumerate(h_samples):
                        value = lane_coords[sample_idx] if sample_idx < len(lane_coords) else None
                        if value is None:
                            continue
                        try:
                            x_value = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(x_value) or x_value < 0:
                            continue
                        y_value = float(sample)
                        if y_value < 0 or y_value >= image_shape[0]:
                            continue
                        x_pixel = int(round(x_value))
                        if x_pixel < 0 or x_pixel >= image_shape[1]:
                            continue
                        if mask[int(round(y_value)), x_pixel] > 0:
                            aligned = True
                            break
                    if aligned:
                        break
                results.append({
                    "raw_file": raw_file,
                    "split": split_name,
                    "mask_path": str(mask_path),
                    "aligned": aligned,
                    "lane_count": len([lane for lane in lanes if isinstance(lane, list)]),
                })
                break

        if not mask_found:
            results.append({"raw_file": raw_file, "mask_found": False})

    return {"checked": len(results), "results": results, "verified": any(item.get("aligned") for item in results if isinstance(item, dict))}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "datasets" / "tusimple" / "TUSimple_prepared"
    source_root = repo_root / "datasets" / "tusimple" / "TUSimple"
    output_dir = repo_root / "backend" / "training" / "outputs" / "tusimple_preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = _collect_dataset_stats(dataset_root, output_dir)

    selected_samples: list[dict[str, Any]] = []
    rng = random.Random(42)
    for split_name in ("train", "val", "test"):
        image_paths = _collect_split_files(dataset_root, split_name)
        if not image_paths:
            continue
        target_count = {"train": 10, "val": 5, "test": 5}[split_name]
        sample_paths = rng.sample(image_paths, k=min(target_count, len(image_paths)))
        for index, image_path in enumerate(sample_paths):
            mask_path = dataset_root / split_name / "masks" / image_path.name
            record = _validate_sample(image_path, mask_path, split_name, index, output_dir)
            selected_samples.append(record)

    alignment = _sample_annotation_alignment(dataset_root, source_root, output_dir, count=10)

    summary = {
        "dataset_root": str(dataset_root),
        "preview_dir": str(output_dir),
        "stats": stats,
        "selected_samples": selected_samples,
        "annotation_alignment": alignment,
    }

    (output_dir / "qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["stats"], indent=2))
    print(json.dumps({"selected_sample_count": len(selected_samples), "preview_count": len(list(output_dir.glob("*_preview.png")))}, indent=2))
    print(json.dumps(alignment, indent=2))


if __name__ == "__main__":
    main()
