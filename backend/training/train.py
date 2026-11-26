"""Train the LaneVision U-Net on a prepared dataset (train/, val/, test/ with images/ and masks/).

Model selection uses the validation split only: the best epoch is the one with the highest validation `monitor`
metric (Dice by default, computed over the whole split), early stopping watches the same metric, and the test split
is read only after training, with the selected weights, when evaluate_test is set. Every run writes to output_dir:

    config.json          resolved configuration, git commit, dataset counts and fingerprints, environment
    metrics.jsonl        one JSON object per epoch (losses, validation metrics, learning rate, time, GPU memory)
    training_history.json  the same rows as a list (kept for existing tooling)
    summary.json         best epoch, stop reason, restored-best validation metrics, optional test metrics
    curves.png           loss, validation metrics and learning rate per epoch (when matplotlib is installed)

and to save_dir: lane_segmentation_best.pth (best validation epoch) and lane_segmentation_last.pth (every epoch, with
optimizer, scheduler, AMP scaler, early-stopping and RNG state, so --resume continues the same run exactly).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from backend.app.ai.models.unet import LaneSegmentationUNet
from backend.training.config import TrainingConfig
from backend.training.dataset import LaneSegmentationDataset
from backend.training.losses import build_loss
from backend.training.metrics import confusion_counts, metrics_from_counts
from backend.training.transforms import build_inference_transform, build_train_transforms
from backend.training.tusimple_benchmark import records_for_images, score_probabilities
from backend.training.utils import ensure_dir, load_checkpoint, restore_rng_state, rng_state, save_checkpoint, seed_worker, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


def _resolve_device(device: str) -> torch.device:
    requested = (device or "auto").strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def check_split_separation(dataset_root: str | Path, content: bool = False) -> dict[str, Any]:
    """Fail if a sample appears in more than one split (by file name, and by image bytes when content=True)."""
    root = Path(dataset_root)
    names = {split: sorted(path.name for path in (root / split / "images").glob("*") if path.is_file()) for split in SPLITS if (root / split / "images").is_dir()}
    report: dict[str, Any] = {split: {"images": len(files), "names_sha1": hashlib.sha1("\n".join(files).encode()).hexdigest()[:12]} for split, files in names.items()}
    splits = list(names)
    for i, first in enumerate(splits):
        for second in splits[i + 1:]:
            shared = set(names[first]) & set(names[second])
            if shared:
                raise ValueError(f"{len(shared)} file names are in both {first} and {second} (e.g. {sorted(shared)[0]})")
    if content:
        seen: dict[str, str] = {}
        for split in splits:
            for name in names[split]:
                digest = hashlib.sha1((root / split / "images" / name).read_bytes()).hexdigest()
                if digest in seen and seen[digest] != split:
                    raise ValueError(f"identical image content in {seen[digest]} and {split} ({name})")
                seen.setdefault(digest, split)
        report["content_checked"] = True
    return report


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        except Exception:
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(status) if status is not None else None}


def _environment(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "device": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(gpu=properties.name, gpu_memory_mb=round(properties.total_memory / 2**20), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return info


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """The learning-rate schedule, stepped once per epoch (plateau: with the validation metric)."""
    if config.scheduler == "none":
        return None
    if config.scheduler == "plateau":
        patience = config.scheduler_patience if config.scheduler_patience is not None else max(1, config.patience // 2)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=config.scheduler_factor)
    warmup = max(0, config.warmup_epochs)
    floor = config.min_learning_rate / config.learning_rate

    def factor(epoch: int) -> float:  # epoch = number of completed epochs
        if epoch < warmup:
            return 0.1 + 0.9 * epoch / warmup
        if config.scheduler == "cosine":
            progress = min(1.0, (epoch - warmup) / max(1, config.epochs - warmup))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.scheduler_factor ** ((epoch - warmup) // max(1, config.step_size))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _loader(dataset, config: TrainingConfig, shuffle: bool, generator: torch.Generator | None, pin_memory: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, pin_memory=pin_memory,
                      persistent_workers=False, worker_init_fn=seed_worker, generator=generator)


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device, config: TrainingConfig, loss_fn=None, tusimple_records=None) -> tuple[float, dict[str, float]]:
    """Validation loss and pixel metrics over the whole split (counts summed over batches, threshold 0.5), plus the
    official TuSimple accuracy / FP / FN when the split's label records are given (in dataset order)."""
    loss_fn = loss_fn or build_loss(config)
    model.eval()
    running_loss = 0.0
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    probabilities = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            running_loss += loss_fn(logits, masks).item() * images.size(0)
            for key, value in confusion_counts(logits, masks).items():
                counts[key] += value
            if tusimple_records is not None:
                probabilities.extend(torch.sigmoid(logits[:, 0].float()).cpu().numpy())
    metrics = metrics_from_counts(counts)
    if tusimple_records is not None:
        metrics.update(score_probabilities(probabilities, tusimple_records))
    return running_loss / max(len(dataloader.dataset), 1), metrics


def plot_curves(history: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Loss")
    for key in ("dice", "iou", "precision", "recall"):
        axes[1].plot(epochs, [row[key] for row in history], label=key)
    axes[1].set_title("Validation pixel metrics (threshold 0.5)")
    rates = [row["learning_rate"] for row in history]
    axes[2].plot(epochs, rates)
    if min(rates) > 0:
        axes[2].set_yscale("log")
    axes[2].set_title("Learning rate")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return True


def train(config: TrainingConfig, resume: str | None = None, check_content: bool = False) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed, config.deterministic)
    device = _resolve_device(config.device)
    amp = bool(config.amp and device.type == "cuda")
    output_dir = ensure_dir(config.output_dir)
    model_dir = ensure_dir(config.save_dir)
    best_path = model_dir / "lane_segmentation_best.pth"
    last_path = model_dir / "lane_segmentation_last.pth"
    production_path = Path(config.production_checkpoint_path) if config.production_checkpoint_path else None
    if production_path is not None and production_path.exists() and not config.overwrite_production:
        raise FileExistsError(f"{production_path} exists; refusing to overwrite a production checkpoint (set overwrite_production to replace it)")

    root = Path(config.dataset_root)
    splits = check_split_separation(root, content=check_content)
    loading_started = time.perf_counter()
    train_dataset = LaneSegmentationDataset(str(root / "train" / "images"), str(root / "train" / "masks"), input_size=config.input_size,
                                            transform=build_train_transforms(config.input_size, mask_dilation=config.train_mask_dilation), preload=config.preload)
    val_dataset = LaneSegmentationDataset(str(root / "val" / "images"), str(root / "val" / "masks"), input_size=config.input_size,
                                          transform=build_inference_transform(config.input_size), preload=config.preload)
    preload_seconds = round(time.perf_counter() - loading_started, 1) if config.preload else None
    # Lane-level validation metric: label records of the val images only (matched by prepared file name).
    val_records = records_for_images(config.tusimple_val_labels, [path.name for path in val_dataset.image_paths]) if config.tusimple_val_labels else None
    generator = torch.Generator()
    train_loader = _loader(train_dataset, config, shuffle=True, generator=generator, pin_memory=device.type == "cuda")
    val_loader = _loader(val_dataset, config, shuffle=False, generator=None, pin_memory=device.type == "cuda")
    print(f"Train samples={len(train_dataset)} Val samples={len(val_dataset)} device={device} amp={amp}", flush=True)

    model = LaneSegmentationUNet(in_channels=3, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = build_loss(config)

    history: list[dict[str, Any]] = []
    best_value, best_epoch, best_metrics = float("-inf"), None, None
    epochs_without_improvement = 0
    stop_reason = "completed all epochs"
    start_epoch = 1
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        saved_size = checkpoint.get("config", {}).get("input_size")
        if saved_size and list(saved_size) != list(config.input_size):
            raise ValueError(f"resume checkpoint was trained at input_size {saved_size}, config has {list(config.input_size)}")
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = list(checkpoint.get("history", []))
        best_value = float(checkpoint.get("best_metric") if checkpoint.get("best_metric") is not None else float("-inf"))
        best_epoch = checkpoint.get("best_epoch")
        best_metrics = checkpoint.get("best_metrics")
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if checkpoint.get("stopped_early"):
            start_epoch = config.epochs + 1
            stop_reason = checkpoint.get("stop_reason", "stopped early")
        print(f"Resuming from {resume}: next epoch {start_epoch}, best {config.monitor} {best_value:.4f} at epoch {best_epoch}", flush=True)

    run_info = {"experiment_name": config.experiment_name, "config": config.to_dict(), "resolved": {"device": str(device), "amp": amp, "preload_seconds": preload_seconds},
                "git": _git_state(), "dataset": splits, "environment": _environment(device), "resumed_from": resume}
    (output_dir / "config.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    metrics_log = (output_dir / "metrics.jsonl").open("a" if resume else "w", encoding="utf-8")

    try:
        for epoch in range(start_epoch, config.epochs + 1):
            started = time.perf_counter()
            generator.manual_seed(config.seed + epoch)  # same shuffling per epoch, with or without a resume
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            running_loss = 0.0
            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(images)
                loss = loss_fn(logits.float(), masks)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {loss_value}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss_value * images.size(0)
            train_seconds = time.perf_counter() - started

            val_loss, val_metrics = validate(model, val_loader, device, config, loss_fn, val_records)
            if not math.isfinite(val_loss):
                raise RuntimeError(f"Non-finite validation loss at epoch {epoch}: {val_loss}")
            learning_rate = optimizer.param_groups[0]["lr"]
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics[config.monitor])
            elif scheduler is not None:
                scheduler.step()

            value = val_metrics[config.monitor]
            improved = value > best_value + config.min_delta
            if improved:
                best_value, best_epoch, best_metrics = value, epoch, val_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            row = {"epoch": epoch, "train_loss": running_loss / max(len(train_dataset), 1), "val_loss": val_loss,
                   **{k: val_metrics[k] for k in ("dice", "iou", "precision", "recall", "f1", "tusimple_accuracy", "tusimple_fp", "tusimple_fn") if k in val_metrics},
                   "learning_rate": learning_rate, "epoch_seconds": round(time.perf_counter() - started, 2), "train_images_per_second": round(len(train_dataset) / max(train_seconds, 1e-9), 2),
                   "improved": improved}
            if device.type == "cuda":
                row["gpu_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                row["gpu_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / 2**20, 1)
            history.append(row)
            metrics_log.write(json.dumps(row) + "\n")
            metrics_log.flush()
            (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

            stop = config.early_stopping and epochs_without_improvement >= config.patience
            if stop:
                stop_reason = f"early stopping: no {config.monitor} improvement > {config.min_delta} for {config.patience} epochs"
            state = {"history": history, "best_epoch": best_epoch, "best_metrics": best_metrics, "epochs_without_improvement": epochs_without_improvement,
                     "rng_state": rng_state(), "scaler_state_dict": scaler.state_dict(), "stopped_early": stop, "stop_reason": stop_reason}
            if improved:  # no RNG state in the best checkpoint, so the app can load it with torch's weights-only loader
                best_state = {key: value for key, value in state.items() if key != "rng_state"}
                save_checkpoint(model, best_path, epoch, val_metrics, optimizer, scheduler=scheduler, best_metric=best_value, config=config.to_dict(), extra=best_state)
            save_checkpoint(model, last_path, epoch, val_metrics, optimizer, scheduler=scheduler, best_metric=best_value, config=config.to_dict(), extra=state)
            print(f"epoch={epoch}/{config.epochs} train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} "
                  f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} lr={learning_rate:.6f} {row['epoch_seconds']:.0f}s"
                  + (f" tusimple_acc={val_metrics['tusimple_accuracy']:.4f} fp={val_metrics['tusimple_fp']:.4f} fn={val_metrics['tusimple_fn']:.4f}" if "tusimple_accuracy" in val_metrics else "")
                  + (f" gpu_peak={row['gpu_peak_allocated_mb']:.0f}MB" if "gpu_peak_allocated_mb" in row else "") + (" *" if improved else ""), flush=True)
            if stop:
                print(stop_reason, flush=True)
                break
    finally:
        metrics_log.close()

    summary: dict[str, Any] = {"experiment_name": config.experiment_name, "best_epoch": best_epoch, "monitor": config.monitor, "best_value": best_value,
                               "best_val_metrics": best_metrics, "epochs_run": len(history), "stop_reason": stop_reason}
    if config.restore_best and best_path.exists():
        load_checkpoint(best_path, model, map_location=device)
        _, restored = validate(model, val_loader, device, config, loss_fn, val_records)
        summary["restored_best_val_metrics"] = restored  # re-measured with the restored weights
    if production_path is not None and best_path.exists():
        # The best epoch's weights and metadata only (no optimizer or RNG state), whatever `model` currently holds.
        best = torch.load(str(best_path), map_location="cpu", weights_only=False)
        keep = ("epoch", "state_dict", "model_state_dict", "metrics", "model_name", "architecture", "best_metric", "config")
        production_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = production_path.with_name(production_path.name + ".tmp")
        torch.save({key: best[key] for key in keep if key in best}, str(temporary))
        temporary.replace(production_path)
        summary["production_checkpoint"] = str(production_path)
    if config.evaluate_test and (root / "test" / "images").is_dir():
        test_dataset = LaneSegmentationDataset(str(root / "test" / "images"), str(root / "test" / "masks"), input_size=config.input_size,
                                               transform=build_inference_transform(config.input_size))
        _, summary["test_metrics_after_selection"] = validate(model, _loader(test_dataset, config, False, None, device.type == "cuda"), device, config, loss_fn)
    summary["curves"] = "curves.png" if history and plot_curves(history, output_dir / "curves.png") else None
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"history": history, "best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
            "production_checkpoint": str(production_path) if production_path else None, "best_metrics": best_metrics or {}, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LaneVision AI U-Net segmentation model")
    parser.add_argument("--config", default=None, help="YAML config (backend/training/configs/)")
    parser.add_argument("--dataset", dest="dataset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="lane_segmentation_last.pth of the run to continue")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config field (value parsed as YAML)")
    parser.add_argument("--check-content", action="store_true", help="also compare image bytes across splits (slower)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
    overrides = {"dataset_root": args.dataset, "epochs": args.epochs, "batch_size": args.batch_size, "device": args.device}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key.strip()] = yaml.safe_load(value)
    for key, value in overrides.items():
        if value is None and key in ("dataset_root", "epochs", "batch_size", "device"):
            continue
        if not hasattr(config, key):
            raise SystemExit(f"Unknown config field: {key}")
        setattr(config, key, tuple(value) if key == "input_size" else value)
    result = train(config.validate(), resume=args.resume, check_content=args.check_content)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
