from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from backend.app.ai.config import AIConfig
from backend.app.ai.model_loader import ModelLoader
from backend.training.config import TrainingConfig
from backend.training.losses import build_loss, tversky_loss
from backend.training.train import build_scheduler, check_split_separation, train
from backend.training.transforms import build_inference_transform, build_train_transforms


def _dataset(root: Path, counts=(6, 3, 3), size=(64, 32)) -> Path:
    rng = np.random.default_rng(0)
    width, height = size
    for split, count in zip(("train", "val", "test"), counts):
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)
        for index in range(count):
            image = rng.integers(0, 60, (height, width, 3), dtype=np.uint8)
            mask = np.zeros((height, width), dtype=np.uint8)
            x = int(rng.integers(8, width - 8))
            cv2.line(image, (x, height - 1), (width // 2, 4), (230, 230, 230), 2)
            cv2.line(mask, (x, height - 1), (width // 2, 4), 255, 1)
            cv2.imwrite(str(root / split / "images" / f"{split}_{index}.png"), image)
            cv2.imwrite(str(root / split / "masks" / f"{split}_{index}.png"), mask)
    return root


def _config(tmp_path: Path, **overrides) -> TrainingConfig:
    config = TrainingConfig(dataset_root=str(tmp_path / "data"), input_size=(64, 32), epochs=3, batch_size=2, learning_rate=1e-3, device="cpu",
                            save_dir=str(tmp_path / "run" / "models"), output_dir=str(tmp_path / "run" / "outputs"), experiment_name="test")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config.validate()


@pytest.fixture()
def data(tmp_path: Path) -> Path:
    return _dataset(tmp_path / "data")


def test_early_stopping_stops_after_patience_and_restores_the_best_epoch(tmp_path, data) -> None:
    result = train(_config(tmp_path, epochs=10, learning_rate=0.0, patience=2, scheduler="none"))  # nothing can improve after epoch 1
    summary = result["summary"]
    assert summary["epochs_run"] == 3
    assert summary["best_epoch"] == 1
    assert summary["stop_reason"].startswith("early stopping")
    assert summary["restored_best_val_metrics"]["dice"] == pytest.approx(summary["best_val_metrics"]["dice"])


def test_run_writes_structured_logs(tmp_path, data) -> None:
    result = train(_config(tmp_path, epochs=2, early_stopping=False))
    outputs = tmp_path / "run" / "outputs"
    rows = [json.loads(line) for line in (outputs / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert {"train_loss", "val_loss", "dice", "iou", "precision", "recall", "learning_rate", "epoch_seconds", "train_images_per_second"} <= set(rows[0])
    run = json.loads((outputs / "config.json").read_text(encoding="utf-8"))
    assert run["config"]["input_size"] == [64, 32] and run["resolved"] == {"device": "cpu", "amp": False}
    assert run["dataset"]["train"]["images"] == 6 and run["dataset"]["test"]["images"] == 3
    assert json.loads((outputs / "summary.json").read_text(encoding="utf-8"))["epochs_run"] == 2
    assert result["summary"]["curves"] in ("curves.png", None)
    if result["summary"]["curves"]:
        assert (outputs / "curves.png").stat().st_size > 0
    assert "test_metrics_after_selection" not in result["summary"]  # the test split is only read when asked


def test_same_seed_gives_the_same_run(tmp_path, data) -> None:
    first = train(_config(tmp_path, epochs=2, output_dir=str(tmp_path / "a"), save_dir=str(tmp_path / "a")))["history"]
    second = train(_config(tmp_path, epochs=2, output_dir=str(tmp_path / "b"), save_dir=str(tmp_path / "b")))["history"]
    assert [row["train_loss"] for row in first] == [row["train_loss"] for row in second]
    assert [row["dice"] for row in first] == [row["dice"] for row in second]


def test_resume_continues_the_same_run(tmp_path, data) -> None:
    # Three epochs in one go, versus two epochs and a resume to three: same losses, learning rates and final weights.
    def config(folder: str, epochs: int) -> TrainingConfig:
        return _config(tmp_path, epochs=epochs, early_stopping=False, scheduler="step", step_size=1, output_dir=str(tmp_path / folder), save_dir=str(tmp_path / folder))

    straight = train(config("straight", 3))
    train(config("resumed", 2))
    resumed = train(config("resumed", 3), resume=str(tmp_path / "resumed" / "lane_segmentation_last.pth"))

    assert [row["epoch"] for row in resumed["history"]] == [1, 2, 3]
    assert [row["train_loss"] for row in resumed["history"]] == pytest.approx([row["train_loss"] for row in straight["history"]], rel=1e-6)
    assert [row["learning_rate"] for row in resumed["history"]] == pytest.approx([row["learning_rate"] for row in straight["history"]])
    weights_a = torch.load(tmp_path / "straight" / "lane_segmentation_last.pth", map_location="cpu", weights_only=False)["state_dict"]
    weights_b = torch.load(tmp_path / "resumed" / "lane_segmentation_last.pth", map_location="cpu", weights_only=False)["state_dict"]
    assert all(torch.allclose(weights_a[key].float(), weights_b[key].float(), atol=1e-6) for key in weights_a)


def test_learning_rate_schedules() -> None:
    def rates(**overrides) -> list[float]:
        config = TrainingConfig(epochs=6, learning_rate=1e-2, **overrides).validate()
        optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=config.learning_rate)
        scheduler = build_scheduler(optimizer, config)
        values = []
        for _ in range(config.epochs):
            values.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return values

    assert rates(scheduler="step", step_size=2, scheduler_factor=0.5) == pytest.approx([1e-2, 1e-2, 5e-3, 5e-3, 2.5e-3, 2.5e-3])
    cosine = rates(scheduler="cosine", warmup_epochs=2, min_learning_rate=1e-4)
    assert cosine[0] == pytest.approx(1e-3) and cosine[2] == pytest.approx(1e-2)  # warm-up from lr/10, then the peak
    assert all(later < earlier for earlier, later in zip(cosine[2:], cosine[3:]))
    assert TrainingConfig(scheduler="none").validate() and build_scheduler(torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0), TrainingConfig(scheduler="none")) is None


@pytest.mark.parametrize("loss", ["bce_dice", "focal_dice", "tversky", "focal_tversky", "bce"])
def test_every_loss_is_finite_and_trainable(loss) -> None:
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    targets = (torch.rand(2, 1, 8, 8) > 0.8).float()
    value = build_loss(TrainingConfig(loss=loss).validate())(logits, targets)
    value.backward()
    assert torch.isfinite(value) and logits.grad is not None and torch.isfinite(logits.grad).all()


def test_tversky_with_beta_above_alpha_penalises_missed_lane_pixels_more() -> None:
    targets = torch.zeros(1, 1, 4, 4)
    targets[0, 0, :, 1] = 1.0
    confident = torch.full((1, 2, 4, 4), -10.0)
    confident[0, 0, :, 1] = 10.0
    missed = confident.clone()
    missed[0, 0, :2, 1] = -10.0  # two lane pixels missed
    extra = confident.clone()
    extra[0, 0, :2, 2] = 10.0  # two background pixels predicted as lane
    assert tversky_loss(missed, targets, alpha=0.3, beta=0.7) > tversky_loss(extra, targets, alpha=0.3, beta=0.7)


def test_mask_dilation_thickens_training_targets_only() -> None:
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    mask = np.zeros((32, 64), dtype=np.uint8)
    mask[:, 30] = 255
    _, plain = build_inference_transform((64, 32))(image, mask)
    _, thick = build_train_transforms((64, 32), mask_dilation=2)(image, mask)
    assert plain.sum().item() == 32
    assert thick.sum().item() == 32 * 5


def test_split_overlap_is_rejected(tmp_path) -> None:
    root = _dataset(tmp_path / "data")
    assert check_split_separation(root, content=True)["content_checked"] is True
    (root / "val" / "images" / "train_0.png").write_bytes((root / "train" / "images" / "train_0.png").read_bytes())
    with pytest.raises(ValueError, match="both train and val"):
        check_split_separation(root)
    (root / "val" / "images" / "train_0.png").unlink()
    (root / "test" / "images" / "copy.png").write_bytes((root / "train" / "images" / "train_1.png").read_bytes())
    with pytest.raises(ValueError, match="identical image content"):
        check_split_separation(root, content=True)


def test_existing_production_checkpoint_is_never_overwritten(tmp_path, data) -> None:
    production = tmp_path / "production.pth"
    production.write_bytes(b"the working model")
    with pytest.raises(FileExistsError):
        train(_config(tmp_path, production_checkpoint_path=str(production)))
    assert production.read_bytes() == b"the working model"


def test_exported_model_is_served_at_its_training_input_size(tmp_path, data) -> None:
    exported = tmp_path / "export" / "lane_segmentation.pth"
    train(_config(tmp_path, epochs=2, early_stopping=False, production_checkpoint_path=str(exported), evaluate_test=True))
    payload = torch.load(exported, map_location="cpu")  # loadable with torch's default (weights-only) loader
    assert payload["config"]["input_size"] == [64, 32] and "optimizer_state_dict" not in payload
    metadata = ModelLoader(AIConfig(enabled=True, model_path=str(exported), device="cpu", input_width=320, input_height=160)).load()
    assert (metadata.input_width, metadata.input_height) == (64, 32)
    best = torch.load(tmp_path / "run" / "models" / "lane_segmentation_best.pth", map_location="cpu")  # also weights-only loadable
    assert best["epoch"] == payload["epoch"]


def test_amp_is_only_enabled_on_cuda(tmp_path, data) -> None:
    train(_config(tmp_path, epochs=1, amp=True))
    assert json.loads((tmp_path / "run" / "outputs" / "config.json").read_text(encoding="utf-8"))["resolved"]["amp"] is False


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(input_size=(300, 160)).validate()  # not a multiple of 16
    with pytest.raises(ValueError):
        TrainingConfig(loss="hinge").validate()
    with pytest.raises(ValueError):
        TrainingConfig(scheduler="cyclic").validate()
