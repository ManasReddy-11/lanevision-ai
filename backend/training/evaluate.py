from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backend.app.ai.models.unet import LaneSegmentationUNet
from backend.training.dataset import LaneSegmentationDataset
from backend.training.metrics import confusion_counts, metrics_from_counts
from backend.training.transforms import build_inference_transform
from backend.training.utils import load_checkpoint


def _resolve_device(device: str) -> torch.device:
    requested = (device or "auto").strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def evaluate(dataset_root: str, checkpoint_path: str, device_name: str = "auto", batch_size: int = 4) -> dict[str, float]:
    """Pixel metrics of a checkpoint on one split folder (images/, masks/), over the whole split at threshold 0.5.

    The input size is the one the checkpoint was trained at (its saved config), 320x160 for checkpoints without it."""
    device = _resolve_device(device_name)
    model = LaneSegmentationUNet(in_channels=3, num_classes=2).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    input_size = tuple(checkpoint.get("config", {}).get("input_size") or (320, 160))
    model.eval()

    dataset = LaneSegmentationDataset(
        image_dir=str(Path(dataset_root) / "images"),
        mask_dir=str(Path(dataset_root) / "masks"),
        input_size=input_size,
        transform=build_inference_transform(input_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    with torch.no_grad():
        for images, masks in dataloader:
            logits = model(images.to(device))
            for key, value in confusion_counts(logits, masks.to(device)).items():
                counts[key] += value

    metrics = metrics_from_counts(counts)
    return {key: metrics[key] for key in ("dice", "iou", "precision", "recall", "f1")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LaneVision segmentation checkpoint")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(evaluate(args.dataset, args.checkpoint, device_name=args.device, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
