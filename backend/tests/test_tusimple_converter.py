from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.training.datasets.tusimple import (
    _resolve_image_path,
    build_split_manifest,
    convert_tusimple_to_dataset,
    discover_tusimple_source_root,
    inspect_tusimple_source,
    validate_dataset,
)


def _split_names(output_root: Path, folder: str = "images") -> dict[str, set[str]]:
    return {split: {path.name for path in (output_root / split / folder).glob("*.png")} for split in ("train", "val", "test")}


def _lane_record(raw_file: str, offset: int = 0) -> dict:
    return {"raw_file": raw_file, "lanes": [[10 + offset, 20, 30], [40, 45, 50]], "h_samples": [10, 30, 50]}


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

    images = _split_names(output_root, "images")
    assert images == _split_names(output_root, "masks")
    assert sum(len(names) for names in images.values()) == 1

    validation = validate_dataset(output_root)
    assert validation["train_count"] + validation["val_count"] + validation["test_count"] == 1
    assert validation["validation_count"] == validation["val_count"]
    assert validation["unpaired_files"] == 0


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
    splits_holding_sample = [split for split, names in _split_names(output_root).items() if "clips_0313-1_6040_20.png" in names]
    assert len(splits_holding_sample) == 1


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
    images = _split_names(output_root)
    assert set().union(*images.values()) == {"clips_0313-1_6040_20.png", "clips_0313-2_6040_20.png"}
    assert sum(len(names) for names in images.values()) == 2


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


def test_tusimple_converter_writes_disjoint_splits_that_cover_every_valid_pair(tmp_path: Path) -> None:
    train_set = tmp_path / "tusimple_source" / "train_set"
    raw_files = [f"clips/0313-1/{index}/20.jpg" for index in range(20)]
    for raw_file in raw_files:
        (train_set / raw_file).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(train_set / raw_file), np.zeros((64, 64, 3), dtype=np.uint8))
    (train_set / "label_data_0313.json").write_text("\n".join(json.dumps(_lane_record(raw)) for raw in raw_files), encoding="utf-8")

    output_root = tmp_path / "prepared"
    manifest = convert_tusimple_to_dataset(tmp_path / "tusimple_source", output_root, split_seed=42)

    images = _split_names(output_root, "images")
    assert images == _split_names(output_root, "masks")
    assert images == {split: set(names) for split, names in manifest["split"].items()}
    assert not images["train"] & images["val"]
    assert not images["train"] & images["test"]
    assert not images["val"] & images["test"]
    assert sum(len(names) for names in images.values()) == manifest["valid_pairs"] == 20
    assert len(images["val"]) > 0 and len(images["test"]) > 0

    validation = validate_dataset(output_root)
    assert validation["train_count"] == len(manifest["split"]["train"])
    assert validation["train_val_overlap"] == validation["train_test_overlap"] == validation["val_test_overlap"] == 0
    assert validation["unpaired_files"] == 0


def test_tusimple_converter_converts_each_image_once_when_annotation_files_repeat_it(tmp_path: Path) -> None:
    source_root = tmp_path / "tusimple_source"
    raw_files = [f"clips/0530/{index}/20.jpg" for index in range(10)]
    for raw_file in raw_files:
        (source_root / raw_file).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(source_root / raw_file), np.zeros((64, 64, 3), dtype=np.uint8))
    (source_root / "test_label.json").write_text("\n".join(json.dumps(_lane_record(raw)) for raw in raw_files), encoding="utf-8")
    repeated = [_lane_record(raw) for raw in raw_files]
    repeated[0] = _lane_record(raw_files[0], offset=5)
    (source_root / "test_label_new.json").write_text("\n".join(json.dumps(record) for record in repeated), encoding="utf-8")

    output_root = tmp_path / "prepared"
    manifest = convert_tusimple_to_dataset(source_root, output_root, split_seed=42)

    assert manifest["total_annotations"] == 20
    assert manifest["valid_pairs"] == 10
    assert manifest["duplicate_references"] == 10
    assert manifest["conflicting_duplicates"] == 1
    split_items = [item for items in manifest["split"].values() for item in items]
    assert len(split_items) == len(set(split_items)) == 10
    validation = validate_dataset(output_root)
    assert validation["train_val_overlap"] == validation["train_test_overlap"] == validation["val_test_overlap"] == 0


def test_tusimple_resolver_never_maps_a_missing_raw_file_to_an_image_that_only_shares_its_basename(tmp_path: Path) -> None:
    source_root = tmp_path / "tusimple_source"
    source_root.mkdir(parents=True)
    cv2.imwrite(str(source_root / "20.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
    # Every TuSimple keyframe is named 20.jpg; a missing clip must stay missing instead of borrowing this image.
    records = [_lane_record("20.jpg"), _lane_record("clips/0601/missing_clip/20.jpg")]
    annotation_path = source_root / "label_data_0601.json"
    annotation_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    assert not _resolve_image_path(source_root, "clips/0601/missing_clip/20.jpg", annotation_path=annotation_path).exists()
    manifest = convert_tusimple_to_dataset(source_root, tmp_path / "prepared", split_seed=42)

    assert manifest["valid_pairs"] == 1
    assert manifest["invalid_missing_images"] == 1
    assert manifest["ambiguous_samples"] == 0


def test_tusimple_converter_refuses_an_output_root_that_already_holds_samples(tmp_path: Path) -> None:
    source_root = tmp_path / "tusimple_source"
    source_root.mkdir(parents=True)
    cv2.imwrite(str(source_root / "sample.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    (source_root / "label_data.json").write_text(json.dumps([_lane_record("sample.png")]), encoding="utf-8")
    output_root = tmp_path / "prepared"
    (output_root / "val" / "images").mkdir(parents=True)
    cv2.imwrite(str(output_root / "val" / "images" / "stale.png"), np.zeros((8, 8, 3), dtype=np.uint8))

    with pytest.raises(FileExistsError):
        convert_tusimple_to_dataset(source_root, output_root)


def test_kaggle_prepare_refuses_to_reuse_a_prepared_dataset_with_split_overlap(tmp_path: Path) -> None:
    from backend.training.kaggle_gpu_prep import prepare

    prepared_root = tmp_path / "prepared"
    for split in ("train", "val"):
        for folder in ("images", "masks"):
            (prepared_root / split / folder).mkdir(parents=True)
            cv2.imwrite(str(prepared_root / split / folder / "clips_0313-1_1_20.png"), np.zeros((8, 8), dtype=np.uint8))
    assert validate_dataset(prepared_root)["train_val_overlap"] == 1

    args = argparse.Namespace(raw_root=str(tmp_path), search_root=str(tmp_path), prepared_root=str(prepared_root), lane_thickness=2, split_seed=42)
    with pytest.raises(RuntimeError, match="shares samples between splits"):
        prepare(args)


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def _write_kaggle_tusimple_layout(dataset_root: Path) -> None:
    """Scaled-down copy of the audited Kaggle layout (manideep1108/tusimple, version 5).

    TUSimple/train_set/{label_data_0313,0531,0601}.json with clips/ next to them; TUSimple/test_label.json next to
    TUSimple/test_set/, whose clips its raw_file values (clips/...) refer to; test_label_new.json at the dataset root
    with the same records spelled test_set/clips/...; test_tasks_0627.json without lane points; seg_label/ copies.
    """
    tusimple = dataset_root / "TUSimple"
    train_files = {"label_data_0313.json": ("0313-1", 4), "label_data_0531.json": ("0531", 2), "label_data_0601.json": ("0601", 2)}
    train_records = []
    for name, (date, count) in train_files.items():
        records = [_lane_record(f"clips/{date}/{index}/20.jpg") for index in range(count)]
        for record in records:
            (tusimple / "train_set" / record["raw_file"]).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tusimple / "train_set" / record["raw_file"]), np.zeros((64, 64, 3), dtype=np.uint8))
        _write_records(tusimple / "train_set" / name, records)
        train_records += records
    test_records = [_lane_record(f"clips/0530/{index}/20.jpg") for index in range(3)]
    for record in test_records:
        (tusimple / "test_set" / record["raw_file"]).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tusimple / "test_set" / record["raw_file"]), np.zeros((64, 64, 3), dtype=np.uint8))
    _write_records(tusimple / "test_label.json", test_records)
    _write_records(dataset_root / "test_label_new.json", [{**record, "raw_file": "test_set/" + record["raw_file"]} for record in test_records])
    _write_records(tusimple / "test_set" / "test_tasks_0627.json", [{**record, "lanes": []} for record in test_records])
    _write_records(tusimple / "train_set" / "seg_label" / "test.json", test_records)
    _write_records(tusimple / "train_set" / "seg_label" / "train_val.json", train_records)


def test_tusimple_discovery_finds_a_dataset_mounted_several_levels_below_the_search_root(tmp_path: Path) -> None:
    search_root = tmp_path / "input"
    dataset_root = search_root / "datasets" / "owner" / "tusimple" / "tusimple"
    _write_kaggle_tusimple_layout(dataset_root)

    discovery = discover_tusimple_source_root(search_root)

    best = discovery["best_match"]
    assert Path(best["source_root"]) == dataset_root
    assert best["sampled_record_count"] > 0
    assert best["resolved_record_count"] == best["sampled_record_count"]


def test_tusimple_converter_resolves_the_audited_kaggle_layout_and_merges_test_label_new(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tusimple"
    _write_kaggle_tusimple_layout(dataset_root)

    manifest = convert_tusimple_to_dataset(dataset_root, tmp_path / "prepared", split_seed=42)

    assert manifest["invalid_missing_images"] == 0
    assert manifest["valid_pairs"] == 8 + 3
    assert manifest["duplicate_references"] == 3
    assert manifest["conflicting_duplicates"] == 0
    assert manifest["ambiguous_samples"] == 0
    assert manifest["empty_masks"] == 3  # test_tasks_0627.json carries no lane points
    validation = validate_dataset(tmp_path / "prepared")
    assert validation["train_val_overlap"] == validation["train_test_overlap"] == validation["val_test_overlap"] == 0


def test_tusimple_converter_official_split_keeps_the_benchmark_files_apart(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tusimple"
    _write_kaggle_tusimple_layout(dataset_root)
    output_root = tmp_path / "prepared"

    manifest = convert_tusimple_to_dataset(dataset_root, output_root, split_mode="official")

    images = _split_names(output_root)
    assert images["train"] == {f"clips_0313-1_{index}_20.png" for index in range(4)} | {f"clips_0601_{index}_20.png" for index in range(2)}
    assert images["val"] == {f"clips_0531_{index}_20.png" for index in range(2)}
    assert images["test"] == {f"clips_0530_{index}_20.png" for index in range(3)}
    assert images == _split_names(output_root, "masks")
    assert [Path(path).name for path in manifest["ignored_annotation_files"]] == ["test_tasks_0627.json"]
    assert manifest["duplicate_references"] == 3


def test_tusimple_converter_official_split_rejects_an_image_annotated_in_two_splits(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tusimple"
    _write_kaggle_tusimple_layout(dataset_root)
    train_file = dataset_root / "TUSimple" / "train_set" / "label_data_0601.json"
    train_file.write_text(train_file.read_text(encoding="utf-8") + "\n" + json.dumps(_lane_record("clips/0530/0/20.jpg")), encoding="utf-8")

    with pytest.raises(ValueError, match="more than one official split"):
        convert_tusimple_to_dataset(dataset_root, tmp_path / "prepared", split_mode="official")


def test_tusimple_converter_official_split_requires_train_val_and_test_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tusimple"
    _write_kaggle_tusimple_layout(dataset_root)
    (dataset_root / "TUSimple" / "test_label.json").unlink()
    (dataset_root / "test_label_new.json").unlink()

    with pytest.raises(FileNotFoundError, match="test"):
        convert_tusimple_to_dataset(dataset_root, tmp_path / "prepared", split_mode="official")
