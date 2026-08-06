from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.training.config import TrainingConfig
from backend.training.dataset import LaneSegmentationDataset
from backend.training.losses import DiceLoss, dice_loss
from backend.training.metrics import compute_metrics
from backend.training.transforms import build_train_transforms, build_inference_transform
from backend.training.utils import save_checkpoint, set_seed
from backend.app.ai.models.unet import LaneSegmentationUNet


def test_dataset_pairs_and_mask_binarization(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    cv2.imwrite(str(image_dir / "sample.png"), image)
    cv2.imwrite(str(mask_dir / "sample.png"), mask)

    dataset = LaneSegmentationDataset(str(image_dir), str(mask_dir), input_size=(16, 16))
    sample_image, sample_mask = dataset[0]

    assert sample_image.shape == (3, 16, 16)
    assert sample_mask.shape == (1, 16, 16)
    assert sample_mask.max().item() in {0.0, 1.0}


def test_invalid_dataset_raises_for_mismatched_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    cv2.imwrite(str(image_dir / "a.png"), np.zeros((16, 16, 3), dtype=np.uint8))
    cv2.imwrite(str(mask_dir / "b.png"), np.zeros((16, 16), dtype=np.uint8))

    try:
        LaneSegmentationDataset(str(image_dir), str(mask_dir), input_size=(16, 16))
    except ValueError as exc:
        assert "pair" in str(exc).lower()
    else:
        raise AssertionError("Expected a ValueError for mismatched dataset files")


def test_dice_loss_and_metrics_are_finite() -> None:
    logits = torch.tensor([[[[0.0, 1.0], [0.5, 0.2]]]], dtype=torch.float32)
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32)

    loss = dice_loss(logits, targets)
    metrics = compute_metrics(logits, targets)

    assert torch.isfinite(loss)
    assert metrics["dice"] >= 0.0
    assert metrics["iou"] >= 0.0
    assert metrics["precision"] >= 0.0
    assert metrics["recall"] >= 0.0
    assert metrics["f1"] >= 0.0


def test_model_forward_shape_matches_inference_contract() -> None:
    model = LaneSegmentationUNet(in_channels=3, num_classes=2)
    sample = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    output = model(sample)

    assert output.shape == (1, 2, 32, 32)


def test_checkpoint_save_and_load_round_trip(tmp_path: Path) -> None:
    set_seed(7)
    model = LaneSegmentationUNet(in_channels=3, num_classes=2)
    checkpoint_path = tmp_path / "model.pth"
    save_checkpoint(model, checkpoint_path, epoch=1, metrics={"dice": 0.5, "iou": 0.4})

    loaded = torch.load(checkpoint_path, map_location="cpu")
    assert "state_dict" in loaded
    assert loaded["state_dict"]["out.weight"].shape[0] == 2


def test_training_transform_matches_inference_preprocess() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    transform = build_train_transforms(input_size=(8, 8))
    transformed_image, transformed_mask = transform(image, mask)

    assert transformed_image.shape == (3, 8, 8)
    assert transformed_mask.shape == (1, 8, 8)

    inference_transform = build_inference_transform(input_size=(8, 8))
    image_tensor, mask_tensor = inference_transform(image, mask)
    assert image_tensor.shape == (3, 8, 8)
    assert mask_tensor.shape == (1, 8, 8)
