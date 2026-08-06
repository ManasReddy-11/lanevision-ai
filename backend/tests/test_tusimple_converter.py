from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.training.datasets.tusimple import (
    build_split_manifest,
    convert_tusimple_to_dataset,
    discover_tusimple_source_root,
    inspect_tusimple_source,
    validate_dataset,
)


def test_tusimple_converter_handles_valid_and_invalid_annotations(tmp_path: Path) -> None:
    source_root = tmp_path / "tusimple_source"
    source_root.mkdir(parents=True)

    image_path = source_root / "sample.png"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    annotation_payload = [
        {
            "raw_file": "sample.png",
            "lanes": [[5, 10, 20, 30], [15, 25, 35, 45]],
            "h_samples": [10, 20, 30, 40],
        },
        {
            "raw_file": "missing.png",
            "lanes": [[1, 2, 3]],
            "h_samples": [10, 20, 30],
        },
        {
            "raw_file": "sample.png",
            "lanes": [[-2, -2, -2], [-2, -2, -2]],
            "h_samples": [10, 20, 30],
        },
    ]
    annotation_file = source_root / "label_data.json"
    annotation_file.write_text(json.dumps(annotation_payload), encoding="utf-8")

    output_root = tmp_path / "prepared"
    manifest = convert_tusimple_to_dataset(source_root, output_root, lane_thickness=2, split_seed=7)

    assert manifest["total_annotations"] == 3
    assert manifest["valid_pairs"] == 1
    assert manifest["invalid_missing_images"] == 1
    assert manifest["invalid_annotations"] == 1
    assert manifest["masks_generated"] == 1
    assert manifest["empty_masks"] == 1

    train_images = list((output_root / "train" / "images").glob("*.png"))
    train_masks = list((output_root / "train" / "masks").glob("*.png"))
    assert len(train_images) == len(train_masks) > 0

    validation = validate_dataset(output_root)
    assert validation["train_count"] >= 1
    assert validation["validation_count"] >= 0
    assert validation["test_count"] >= 0


def test_tusimple_converter_resolves_images_from_the_official_train_set_layout(tmp_path: Path) -> None:
    source_root = tmp_path / "tusimple_source"
    source_root.mkdir(parents=True)
    train_set = source_root / "train_set"
    train_set.mkdir(parents=True)
    (train_set / "clips" / "0313-1" / "6040").mkdir(parents=True)
    image_path = train_set / "clips" / "0313-1" / "6040" / "20.jpg"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    annotation_path = train_set / "label_data_0313.json"
    annotation_path.write_text(
        json.dumps([{"raw_file": "clips/0313-1/6040/20.jpg", "lanes": [[10, 20, 30], [15, 25, 35]], "h_samples": [10, 20, 30]}]),
        encoding="utf-8",
    )

    output_root = tmp_path / "prepared"
    manifest = convert_tusimple_to_dataset(source_root, output_root, lane_thickness=2, split_seed=3)

    assert manifest["valid_pairs"] == 1
    assert manifest["invalid_missing_images"] == 0
    assert (output_root / "train" / "images" / "clips_0313-1_6040_20.png").exists()


def test_tusimple_converter_keeps_distinct_samples_with_different_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "tusimple_source"
    source_root.mkdir(parents=True)
    train_set = source_root / "train_set"
    train_set.mkdir(parents=True)
    (train_set / "clips" / "0313-1" / "6040").mkdir(parents=True)
    (train_set / "clips" / "0313-2" / "6040").mkdir(parents=True)
    for relative_path in ["clips/0313-1/6040/20.jpg", "clips/0313-2/6040/20.jpg"]:
        image_path = train_set / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))

    annotation_path = train_set / "label_data_0313.json"
    annotation_path.write_text(
        json.dumps([
            {"raw_file": "clips/0313-1/6040/20.jpg", "lanes": [[10, 20, 30]], "h_samples": [10, 20, 30]},
            {"raw_file": "clips/0313-2/6040/20.jpg", "lanes": [[10, 20, 30]], "h_samples": [10, 20, 30]},
        ]),
        encoding="utf-8",
    )

    output_root = tmp_path / "prepared"
    manifest = convert_tusimple_to_dataset(source_root, output_root, lane_thickness=2, split_seed=3)

    assert manifest["valid_pairs"] == 2
    assert len(list((output_root / "train" / "images").glob("*.png"))) == 2


def test_tusimple_source_discovery_finds_nested_kaggle_style_root(tmp_path: Path) -> None:
    input_root = tmp_path / "kaggle_input"
    dataset_root = input_root / "tusimple-raw"
    train_set = dataset_root / "train_set"
    clips_dir = train_set / "clips" / "0313-1" / "6040"
    clips_dir.mkdir(parents=True)
    image_path = clips_dir / "20.jpg"
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))

    annotation_path = train_set / "label_data_0313.json"
    annotation_path.write_text(
        json.dumps([{"raw_file": "clips/0313-1/6040/20.jpg", "lanes": [[10, 20, 30]], "h_samples": [10, 20, 30]}]),
        encoding="utf-8",
    )

    discovery = discover_tusimple_source_root(input_root)
    assert Path(discovery["best_match"]["source_root"]) == dataset_root
    assert discovery["best_match"]["resolved_record_count"] >= 1

    inspection = inspect_tusimple_source(dataset_root)
    assert inspection["annotation_files"]
    assert inspection["sample_records"][0]["exists"] is True
