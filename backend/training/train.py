from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from backend.app.ai.models.unet import LaneSegmentationUNet
from backend.training.config import TrainingConfig
from backend.training.dataset import LaneSegmentationDataset
from backend.training.losses import total_loss
from backend.training.metrics import compute_metrics
from backend.training.transforms import build_train_transforms
from backend.training.utils import ensure_dir, load_checkpoint, save_checkpoint, set_seed


def _resolve_device(device: str) -> torch.device:
    requested = (device or "auto").strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _make_loader(
    dataset: LaneSegmentationDataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    use_pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=False,
    )


def train(config: TrainingConfig, resume: str | None = None) -> dict[str, Any]:
    set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = ensure_dir(config.output_dir)
    model_dir = ensure_dir(config.save_dir)
    use_pin_memory = device.type == "cuda"

    print("Initializing train dataset...")
    train_dataset = LaneSegmentationDataset(
        image_dir=str(Path(config.dataset_root) / "train" / "images"),
        mask_dir=str(Path(config.dataset_root) / "train" / "masks"),
        input_size=config.input_size,
        transform=build_train_transforms(config.input_size),
    )
    print("Initializing validation dataset...")
    val_dataset = LaneSegmentationDataset(
        image_dir=str(Path(config.dataset_root) / "val" / "images"),
        mask_dir=str(Path(config.dataset_root) / "val" / "masks"),
        input_size=config.input_size,
        transform=build_train_transforms(config.input_size),
    )

    print("Building data loaders...")
    train_loader = _make_loader(
        train_dataset,
        config.batch_size,
        config.num_workers,
        shuffle=True,
        use_pin_memory=use_pin_memory,
    )
    val_loader = _make_loader(
        val_dataset,
        config.batch_size,
        config.num_workers,
        shuffle=False,
        use_pin_memory=use_pin_memory,
    )
    print(f"Train samples={len(train_dataset)} Val samples={len(val_dataset)}")

    model = LaneSegmentationUNet(in_channels=3, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=max(1, config.patience // 2), factor=0.5)

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, float] | None = None
    best_dice = float("-inf")
    start_epoch = 1
    best_path = model_dir / "lane_segmentation_best.pth"
    last_path = model_dir / "lane_segmentation_last.pth"
    production_path = Path(config.production_checkpoint_path)
    production_path.parent.mkdir(parents=True, exist_ok=True)

    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metrics = checkpoint.get("metrics") if isinstance(checkpoint.get("metrics"), dict) else None
        best_dice = float(checkpoint.get("best_metric", best_metrics.get("dice", float("-inf")) if best_metrics else float("-inf")))
        print(f"Resuming from checkpoint={resume} next_epoch={start_epoch} best_dice={best_dice:.4f}")

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = total_loss(logits, masks, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(train_dataset), 1)
        val_loss, val_metrics = validate(model, val_loader, device, config)
        scheduler.step(val_metrics["dice"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dice": val_metrics["dice"],
                "iou": val_metrics["iou"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "f1": val_metrics["f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_metrics = val_metrics
            save_checkpoint(
                model,
                best_path,
                epoch,
                val_metrics,
                optimizer,
                scheduler=scheduler,
                best_metric=best_dice,
                config=config.to_dict(),
            )
            save_checkpoint(
                model,
                production_path,
                epoch,
                val_metrics,
                optimizer,
                scheduler=scheduler,
                best_metric=best_dice,
                config=config.to_dict(),
            )
        save_checkpoint(
            model,
            last_path,
            epoch,
            val_metrics,
            optimizer,
            scheduler=scheduler,
            best_metric=best_dice,
            config=config.to_dict(),
        )

        print(f"epoch={epoch}/{config.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return {
        "history": history,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "production_checkpoint": str(production_path),
        "best_metrics": best_metrics or {},
    }


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device, config: TrainingConfig) -> tuple[float, dict[str, float]]:
    model.eval()
    running_loss = 0.0
    all_metrics: dict[str, list[float]] = {"dice": [], "iou": [], "precision": [], "recall": [], "f1": []}
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = total_loss(logits, masks, bce_weight=config.bce_weight, dice_weight=config.dice_weight)
            running_loss += loss.item() * images.size(0)
            metrics = compute_metrics(logits, masks)
            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key].append(value)

    average_metrics = {key: sum(values) / max(len(values), 1) for key, values in all_metrics.items()}
    return running_loss / max(len(dataloader.dataset), 1), average_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LaneVision AI U-Net segmentation model")
    parser.add_argument("--dataset", dest="dataset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    if args.config:
        config = TrainingConfig.load(args.config)
    if args.dataset is not None:
        config.dataset_root = args.dataset
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
