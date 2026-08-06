from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from backend.app.ai.models.unet import LaneSegmentationUNet
from backend.training.config import TrainingConfig
from backend.training.dataset import LaneSegmentationDataset
from backend.training.datasets.tusimple import convert_tusimple_to_dataset, discover_tusimple_source_root, validate_dataset
from backend.training.losses import total_loss
from backend.training.metrics import compute_metrics
from backend.training.transforms import build_train_transforms
from backend.training.utils import ensure_dir, load_checkpoint, save_checkpoint, set_seed


def _resolve_device(device_name: str) -> torch.device:
    requested = (device_name or "auto").strip().lower()
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepared_dataset_counts(dataset_root: Path) -> dict[str, int]:
    counts = validate_dataset(dataset_root)
    return {
        "train": int(counts.get("train_count", 0)),
        "val": int(counts.get("val_count", counts.get("validation_count", 0))),
        "test": int(counts.get("test_count", 0)),
    }


def _make_loader(dataset: LaneSegmentationDataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def _load_dataset_splits(dataset_root: Path, input_size: tuple[int, int], batch_size: int, device: torch.device) -> tuple[LaneSegmentationDataset, LaneSegmentationDataset, DataLoader, DataLoader]:
    transform = build_train_transforms(input_size)
    train_dataset = LaneSegmentationDataset(
        image_dir=str(dataset_root / "train" / "images"),
        mask_dir=str(dataset_root / "train" / "masks"),
        input_size=input_size,
        transform=transform,
    )
    val_dataset = LaneSegmentationDataset(
        image_dir=str(dataset_root / "val" / "images"),
        mask_dir=str(dataset_root / "val" / "masks"),
        input_size=input_size,
        transform=transform,
    )
    train_loader = _make_loader(train_dataset, batch_size=batch_size, shuffle=True, device=device)
    val_loader = _make_loader(val_dataset, batch_size=batch_size, shuffle=False, device=device)
    return train_dataset, val_dataset, train_loader, val_loader


def _disk_free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(str(target))
    return int(usage.free)


def discover(args: argparse.Namespace) -> dict[str, Any]:
    return discover_tusimple_source_root(args.search_root, sample_size=args.sample_size)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = Path(args.raw_root) if args.raw_root else Path(discover_tusimple_source_root(args.search_root)["best_match"]["source_root"])
    prepared_root = Path(args.prepared_root)
    if prepared_root.exists() and (prepared_root / "train" / "images").exists():
        return {
            "prepared_root": str(prepared_root),
            "raw_root": str(raw_root),
            "prepared_existing": True,
            "counts": _prepared_dataset_counts(prepared_root),
        }

    manifest = convert_tusimple_to_dataset(raw_root, prepared_root, lane_thickness=args.lane_thickness, split_seed=args.split_seed)
    return {
        "prepared_root": str(prepared_root),
        "raw_root": str(raw_root),
        "prepared_existing": False,
        "manifest": manifest,
        "counts": _prepared_dataset_counts(prepared_root),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is unavailable. Kaggle GPU preflight requires a CUDA-enabled runtime.")

    prepared_root = Path(args.prepared_root)
    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {prepared_root}")

    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    counts = _prepared_dataset_counts(prepared_root)
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_device_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "selected_training_device": device.type,
        "dataset_path": str(prepared_root),
        "train_count": counts["train"],
        "val_count": counts["val"],
        "test_count": counts["test"],
        "checkpoint_output_directory": str(checkpoint_dir),
        "free_disk_space_bytes": _disk_free_bytes(checkpoint_dir),
    }


def smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is unavailable. GPU smoke test requires a CUDA-enabled runtime.")

    dataset_root = Path(args.prepared_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_root}")

    set_seed(42)
    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
    config.dataset_root = str(dataset_root)
    config.batch_size = args.batch_size
    config.device = device.type
    config.save_dir = args.checkpoint_dir
    config.production_checkpoint_path = str(Path(args.checkpoint_dir) / "lane_segmentation.pth")
    train_dataset, val_dataset, train_loader, val_loader = _load_dataset_splits(dataset_root, config.input_size, args.batch_size, device)

    model = LaneSegmentationUNet(in_channels=3, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_losses: list[float] = []
    batch_times: list[float] = []
    max_memory_bytes = 0
    completed_batches = 0

    model.train()
    for images, masks in train_loader:
        start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = total_loss(logits, masks, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss during GPU smoke test")
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            max_memory_bytes = max(max_memory_bytes, int(torch.cuda.max_memory_allocated(device=device)))
        batch_times.append(time.perf_counter() - start)
        train_losses.append(float(loss.item()))
        completed_batches += 1
        if completed_batches >= args.train_batches:
            break

    model.eval()
    val_losses: list[float] = []
    dice_values: list[float] = []
    iou_values: list[float] = []
    validation_batches = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = total_loss(logits, masks, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
            metrics = compute_metrics(logits, masks)
            val_losses.append(float(loss.item()))
            dice_values.append(metrics["dice"])
            iou_values.append(metrics["iou"])
            validation_batches += 1
            if validation_batches >= args.val_batches:
                break

    seconds_per_batch = sum(batch_times) / max(len(batch_times), 1)
    full_epoch_batches = math.ceil(len(train_dataset) / args.batch_size)
    minutes_per_epoch = (seconds_per_batch * full_epoch_batches) / 60.0
    return {
        "device": device.type,
        "dataset_path": str(dataset_root),
        "train_batches_completed": completed_batches,
        "val_batches_completed": validation_batches,
        "batch_size": args.batch_size,
        "avg_train_loss": sum(train_losses) / max(len(train_losses), 1),
        "avg_val_loss": sum(val_losses) / max(len(val_losses), 1),
        "avg_val_dice": sum(dice_values) / max(len(dice_values), 1),
        "avg_val_iou": sum(iou_values) / max(len(iou_values), 1),
        "seconds_per_batch": seconds_per_batch,
        "minutes_per_epoch_estimate": minutes_per_epoch,
        "hours_for_30_epochs_estimate": (minutes_per_epoch * 30.0) / 60.0,
        "max_gpu_memory_bytes": max_memory_bytes,
        "checkpoint_output_directory": str(Path(args.checkpoint_dir)),
        "counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
        },
    }


def verify_checkpoint_paths(args: argparse.Namespace) -> dict[str, Any]:
    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
    config.save_dir = args.checkpoint_dir
    config.production_checkpoint_path = str(Path(args.checkpoint_dir) / "lane_segmentation.pth")

    best_path = Path(config.save_dir) / "lane_segmentation_best.pth"
    last_path = Path(config.save_dir) / "lane_segmentation_last.pth"
    production_path = Path(config.production_checkpoint_path)

    tmp_model = LaneSegmentationUNet(in_channels=3, num_classes=2)
    optimizer = torch.optim.AdamW(tmp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    save_checkpoint(
        tmp_model,
        last_path,
        epoch=1,
        metrics={"dice": 0.1},
        optimizer=optimizer,
        scheduler=scheduler,
        best_metric=0.1,
        config=config.to_dict(),
    )
    load_checkpoint(last_path, LaneSegmentationUNet(in_channels=3, num_classes=2), map_location="cpu")
    last_path.unlink(missing_ok=True)

    return {
        "save_dir": str(Path(config.save_dir)),
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "production_checkpoint_path": str(production_path),
        "checkpoint_parent_writable": os.access(str(Path(config.save_dir)), os.W_OK) if Path(config.save_dir).exists() else os.access(str(Path(config.save_dir).parent), os.W_OK),
        "resume_roundtrip_passed": True,
        "checkpoint_frequency": "epoch_end",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle GPU preparation helpers for LaneVision AI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="Discover a TuSimple raw dataset under a search root")
    discover_parser.add_argument("--search-root", default="/kaggle/input")
    discover_parser.add_argument("--sample-size", type=int, default=8)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare TuSimple data into a writable dataset root")
    prepare_parser.add_argument("--search-root", default="/kaggle/input")
    prepare_parser.add_argument("--raw-root", default=None)
    prepare_parser.add_argument("--prepared-root", default="/kaggle/working/tusimple_prepared")
    prepare_parser.add_argument("--lane-thickness", type=int, default=2)
    prepare_parser.add_argument("--split-seed", type=int, default=42)

    preflight_parser = subparsers.add_parser("preflight", help="Run a CUDA-only Kaggle preflight diagnostic")
    preflight_parser.add_argument("--prepared-root", default="/kaggle/working/tusimple_prepared")
    preflight_parser.add_argument("--checkpoint-dir", default="/kaggle/working/models")
    preflight_parser.add_argument("--device", default="auto")

    smoke_parser = subparsers.add_parser("smoke-test", help="Run a short CUDA smoke test on real TuSimple data")
    smoke_parser.add_argument("--prepared-root", default="/kaggle/working/tusimple_prepared")
    smoke_parser.add_argument("--checkpoint-dir", default="/kaggle/working/models")
    smoke_parser.add_argument("--config", default="backend/training/configs/kaggle.yaml")
    smoke_parser.add_argument("--device", default="auto")
    smoke_parser.add_argument("--batch-size", type=int, default=2)
    smoke_parser.add_argument("--train-batches", type=int, default=20)
    smoke_parser.add_argument("--val-batches", type=int, default=10)

    checkpoint_parser = subparsers.add_parser("verify-checkpoints", help="Verify Kaggle checkpoint paths and resume round-trip")
    checkpoint_parser.add_argument("--checkpoint-dir", default="/kaggle/working/models")
    checkpoint_parser.add_argument("--config", default="backend/training/configs/kaggle.yaml")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    handlers = {
        "discover": discover,
        "prepare": prepare,
        "preflight": preflight,
        "smoke-test": smoke_test,
        "verify-checkpoints": verify_checkpoint_paths,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
