from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    epoch: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    best_metric: float | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "model_name": "lane_segmentation_unet",
        "architecture": "u-net",
        "best_metric": best_metric,
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location=map_location)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model weights")
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
