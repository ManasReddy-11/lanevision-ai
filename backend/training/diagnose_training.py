from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
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
from backend.training.losses import total_loss
from backend.training.metrics import compute_metrics
from backend.training.transforms import build_train_transforms
from backend.training.utils import ensure_dir, set_seed

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


@dataclass(slots=True)
class CheckpointInspection:
    found: bool
    path: str | None
    size_bytes: int | None
    epoch: int | None
    state_valid: bool
    resumable: bool
    error: str | None


@dataclass(slots=True)
class SingleBatchResult:
    batch_size: int
    image_shape: list[int]
    mask_shape: list[int]
    image_dtype: str
    mask_dtype: str
    ram_usage_mb: float | None
    forward_pass: str
    loss_finite: bool
    backward_pass: str
    optimizer_step: str
    loss: float | None
    error: str | None


@dataclass(slots=True)
class DiagnosticResult:
    batches_completed: int
    validation_batches_completed: int
    train_loss: float | None
    val_loss: float | None
    val_dice: float | None
    val_iou: float | None
    seconds_per_batch: float | None
    estimated_full_epoch_seconds: float | None
    estimated_30_epochs_seconds: float | None
    ram_usage_mb: float | None
    error: str | None


def get_ram_usage_mb() -> float | None:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def resolve_dataset_root() -> Path:
    candidates = [
        REPO_ROOT / "datasets" / "tusimple" / "TUSimple_prepared",
        REPO_ROOT / "dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Prepared TuSimple dataset root not found")


def count_pairs(dataset_root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        images = list((dataset_root / split / "images").glob("*.png"))
        masks = list((dataset_root / split / "masks").glob("*.png"))
        counts[split] = {"images": len(images), "masks": len(masks)}
    return counts


def inspect_checkpoint() -> CheckpointInspection:
    candidate_paths = [
        REPO_ROOT / "backend" / "models" / "lane_segmentation.pth",
        REPO_ROOT / "backend" / "models" / "lane_segmentation_best.pth",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = torch.load(str(path), map_location="cpu")
            state_dict = payload.get("state_dict") if isinstance(payload, dict) else payload
            epoch = payload.get("epoch") if isinstance(payload, dict) else None
            model = LaneSegmentationUNet(in_channels=3, num_classes=2)
            model.load_state_dict(state_dict)
            resumable = isinstance(payload, dict) and "optimizer_state_dict" in payload
            return CheckpointInspection(
                found=True,
                path=str(path),
                size_bytes=path.stat().st_size,
                epoch=epoch if isinstance(epoch, int) else None,
                state_valid=True,
                resumable=resumable,
                error=None,
            )
        except Exception as exc:
            return CheckpointInspection(
                found=True,
                path=str(path),
                size_bytes=path.stat().st_size,
                epoch=None,
                state_valid=False,
                resumable=False,
                error=str(exc),
            )
    return CheckpointInspection(False, None, None, None, False, False, None)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset: LaneSegmentationDataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def make_datasets(dataset_root: Path, input_size: tuple[int, int]) -> tuple[LaneSegmentationDataset, LaneSegmentationDataset]:
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
    return train_dataset, val_dataset


def run_single_batch_smoke_test(dataset_root: Path, device: torch.device) -> SingleBatchResult:
    try:
        train_dataset, _ = make_datasets(dataset_root, (320, 160))
        loader = make_loader(train_dataset, batch_size=2, shuffle=True, device=device)
        images, masks = next(iter(loader))
        model = LaneSegmentationUNet(in_channels=3, num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = total_loss(logits, masks, bce_weight=0.7, dice_weight=0.3)
        loss_is_finite = bool(torch.isfinite(loss).item())
        if not loss_is_finite:
            raise RuntimeError("Loss became non-finite during smoke test")
        loss.backward()
        optimizer.step()
        return SingleBatchResult(
            batch_size=int(images.shape[0]),
            image_shape=list(images.shape),
            mask_shape=list(masks.shape),
            image_dtype=str(images.dtype),
            mask_dtype=str(masks.dtype),
            ram_usage_mb=get_ram_usage_mb(),
            forward_pass="PASS",
            loss_finite=True,
            backward_pass="PASS",
            optimizer_step="PASS",
            loss=float(loss.item()),
            error=None,
        )
    except Exception as exc:
        return SingleBatchResult(
            batch_size=0,
            image_shape=[],
            mask_shape=[],
            image_dtype="",
            mask_dtype="",
            ram_usage_mb=get_ram_usage_mb(),
            forward_pass="FAIL",
            loss_finite=False,
            backward_pass="FAIL",
            optimizer_step="FAIL",
            loss=None,
            error=str(exc),
        )


def run_short_diagnostic(dataset_root: Path, device: torch.device) -> DiagnosticResult:
    try:
        train_dataset, val_dataset = make_datasets(dataset_root, (320, 160))
        train_loader = make_loader(train_dataset, batch_size=2, shuffle=True, device=device)
        val_loader = make_loader(val_dataset, batch_size=2, shuffle=False, device=device)

        model = LaneSegmentationUNet(in_channels=3, num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        batch_limit = 100
        val_batch_limit = 20
        total_train_loss = 0.0
        train_batches = 0
        batch_times: list[float] = []

        model.train()
        for images, masks in train_loader:
            start = time.perf_counter()
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = total_loss(logits, masks, bce_weight=0.7, dice_weight=0.3)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss during short diagnostic")
            loss.backward()
            optimizer.step()
            batch_times.append(time.perf_counter() - start)
            total_train_loss += float(loss.item())
            train_batches += 1
            if train_batches >= batch_limit:
                break

        model.eval()
        total_val_loss = 0.0
        metric_values: dict[str, list[float]] = {"dice": [], "iou": []}
        val_batches = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = total_loss(logits, masks, bce_weight=0.7, dice_weight=0.3)
                metrics = compute_metrics(logits, masks)
                total_val_loss += float(loss.item())
                metric_values["dice"].append(metrics["dice"])
                metric_values["iou"].append(metrics["iou"])
                val_batches += 1
                if val_batches >= val_batch_limit:
                    break

        seconds_per_batch = sum(batch_times) / len(batch_times)
        full_epoch_batches = math.ceil(len(train_dataset) / 2)
        estimated_full_epoch_seconds = seconds_per_batch * full_epoch_batches
        estimated_30_epochs_seconds = estimated_full_epoch_seconds * 30

        return DiagnosticResult(
            batches_completed=train_batches,
            validation_batches_completed=val_batches,
            train_loss=total_train_loss / max(train_batches, 1),
            val_loss=total_val_loss / max(val_batches, 1),
            val_dice=sum(metric_values["dice"]) / max(len(metric_values["dice"]), 1),
            val_iou=sum(metric_values["iou"]) / max(len(metric_values["iou"]), 1),
            seconds_per_batch=seconds_per_batch,
            estimated_full_epoch_seconds=estimated_full_epoch_seconds,
            estimated_30_epochs_seconds=estimated_30_epochs_seconds,
            ram_usage_mb=get_ram_usage_mb(),
            error=None,
        )
    except Exception as exc:
        return DiagnosticResult(0, 0, None, None, None, None, None, None, None, get_ram_usage_mb(), str(exc))


def main() -> None:
    set_seed(42)
    dataset_root = resolve_dataset_root()
    counts = count_pairs(dataset_root)
    checkpoint = inspect_checkpoint()
    device = resolve_device()
    cpu_name = platform.processor() or "Unknown"
    ram_mb = round((psutil.virtual_memory().total / (1024 * 1024)), 2) if psutil is not None else None

    single_batch = run_single_batch_smoke_test(dataset_root, device)
    diagnostic = None
    if single_batch.forward_pass == "PASS" and single_batch.backward_pass == "PASS" and single_batch.optimizer_step == "PASS":
        diagnostic = run_short_diagnostic(dataset_root, device)
    else:
        diagnostic = DiagnosticResult(0, 0, None, None, None, None, None, None, None, get_ram_usage_mb(), "Skipped because single batch smoke test failed")

    output_dir = ensure_dir(REPO_ROOT / "backend" / "training" / "outputs" / "diagnostic")
    report = {
        "checkpoint": asdict(checkpoint),
        "hardware": {
            "cpu": cpu_name,
            "system_ram_mb": ram_mb,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "training_device": device.type,
        },
        "dataset_root": str(dataset_root),
        "counts": counts,
        "dataloader": {
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": device.type == "cuda",
        },
        "single_batch": asdict(single_batch),
        "short_diagnostic": asdict(diagnostic),
    }
    report_path = output_dir / "crash_recovery_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
