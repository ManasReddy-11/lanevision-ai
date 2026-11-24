from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from backend.app.ai.lane_geometry import LaneCurve
from backend.app.ai.models.unet import LaneSegmentationUNet
from backend.training.tusimple_benchmark import bench, lanes_to_tusimple, line_accuracy, run, summarize

Y = list(range(160, 720, 10))  # 56 h_samples, as in the TuSimple test set


def vertical(x: float) -> list[float]:
    return [x] * len(Y)


def test_perfect_prediction_scores_full_accuracy() -> None:
    gt = [vertical(300), vertical(700)]
    assert bench(gt, gt, Y, 10.0) == (1.0, 0.0, 0.0)


def test_points_beyond_20_px_do_not_match_on_a_vertical_lane() -> None:
    assert line_accuracy(vertical(325), vertical(300), 20.0) == 0.0
    assert line_accuracy(vertical(319), vertical(300), 20.0) == 1.0


def test_a_missing_lane_is_a_false_negative_and_halves_accuracy() -> None:
    accuracy, fp, fn = bench([vertical(300)], [vertical(300), vertical(700)], Y, 10.0)
    assert (accuracy, fp, fn) == (0.5, 0.0, 0.5)


def test_an_extra_lane_is_a_false_positive() -> None:
    accuracy, fp, fn = bench([vertical(300), vertical(900)], [vertical(300)], Y, 10.0)
    assert (accuracy, fp, fn) == (1.0, 0.5, 0.0)


def test_too_many_lanes_or_too_slow_scores_zero() -> None:
    gt = [vertical(300)]
    assert bench([vertical(300)] * 4, gt, Y, 10.0) == (0.0, 0.0, 1.0)
    assert bench(gt, gt, Y, 201.0) == (0.0, 0.0, 1.0)


def test_with_five_ground_truth_lanes_the_worst_lane_is_forgiven() -> None:
    gt = [vertical(x) for x in (100, 350, 600, 850, 1100)]
    accuracy, fp, fn = bench(gt[:4], gt, Y, 10.0)
    assert (accuracy, fp, fn) == (1.0, 0.0, 0.0)


def test_the_threshold_widens_with_the_lane_angle() -> None:
    gt = [float(y) for y in Y]  # x = y: a 45-degree lane, threshold 20 / cos(45) = 28.3 px
    shifted = [x + 25.0 for x in gt]
    assert bench([shifted], [gt], Y, 10.0) == (1.0, 0.0, 0.0)


def test_rows_without_a_lane_in_both_prediction_and_truth_match() -> None:
    gt = [-2] * 20 + [500.0] * (len(Y) - 20)
    assert line_accuracy(gt, gt, 20.0) == 1.0
    assert line_accuracy([-2] * len(Y), gt, 20.0) == pytest.approx(20 / len(Y))


def test_predicted_lanes_are_sampled_only_where_observed_and_inside_the_image() -> None:
    lane = LaneCurve(coefficients=np.array([2.0, -400.0]), points=np.array([[x, y] for y in (400, 500, 600) for x in [2.0 * y - 400.0]]), mean_probability=0.9)
    (row,) = lanes_to_tusimple([lane], [300, 400, 500, 600, 700], width=900)
    assert row == [-2, 400.0, 600.0, 800.0, -2]
    (clipped,) = lanes_to_tusimple([lane], [400, 500, 600], width=700)
    assert clipped == [400.0, 600.0, -2]
    (extended,) = lanes_to_tusimple([lane], [300, 400, 500, 600, 700, 800], width=1150, extend_bottom=True)
    assert extended == [-2, 400.0, 600.0, 800.0, 1000.0, -2]  # continued below the last observed row until it leaves the image


def test_run_scores_a_checkpoint_end_to_end(tmp_path: Path) -> None:
    torch.manual_seed(0)
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": LaneSegmentationUNet(in_channels=3, num_classes=2).state_dict(), "epoch": 1}, checkpoint)
    root = tmp_path / "data"
    records = []
    for index in range(2):
        raw = f"clips/0530/{index}/20.jpg"
        (root / "test_set" / raw).parent.mkdir(parents=True)
        cv2.imwrite(str(root / "test_set" / raw), np.full((720, 1280, 3), 60 + 40 * index, dtype=np.uint8))
        records.append({"raw_file": raw, "lanes": [vertical(400), vertical(900)], "h_samples": Y})
    labels = root / "test_label.json"
    labels.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    args = argparse.Namespace(checkpoint=str(checkpoint), labels=[str(labels)], source_root=str(root), out_dir=str(tmp_path / "out"),
                              only_names=None, device="cpu", geometry=None, extend_bottom=True, examples=1, limit=0, progress=0)
    summary = run(args)

    assert summary["images"] == 2
    assert 0.0 <= summary["tusimple"]["accuracy"] <= 1.0
    assert set(summary["pixel"]) == {"dice", "iou", "precision", "recall"}
    assert summary["checkpoint"]["epoch"] == 1
    assert summary["pixel_metric_threshold"] == 0.5  # same threshold as training metrics, independent of lane extraction
    predictions = [json.loads(line) for line in (tmp_path / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [p["raw_file"] for p in predictions] == [r["raw_file"] for r in records]
    assert all(len(lane) == len(Y) for p in predictions for lane in p["lanes"])
    assert summarize([json.loads(line) for line in (tmp_path / "out" / "per_image.jsonl").read_text(encoding="utf-8").splitlines()])["images"] == 2
