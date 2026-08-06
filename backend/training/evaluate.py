from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backend.app.ai.models.unet import LaneSegmentationUNet
from backend.training.dataset import LaneSegmentationDataset
from backend.training.metrics import compute_metrics
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
    device = _resolve_device(device_name)
    model = LaneSegmentationUNet(in_channels=3, num_classes=2).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    dataset = LaneSegmentationDataset(
        image_dir=str(Path(dataset_root) / "images"),
        mask_dir=str(Path(dataset_root) / "masks"),
        input_size=(320, 160),
        transform=build_inference_transform((320, 160)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    metrics_values = {"dice": [], "iou": [], "precision": [], "recall": [], "f1": []}
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            metrics = compute_metrics(logits, masks)
            for key, value in metrics.items():
                if key in metrics_values:
                    metrics_values[key].append(value)

    return {key: sum(values) / max(len(values), 1) for key, values in metrics_values.items()}


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
