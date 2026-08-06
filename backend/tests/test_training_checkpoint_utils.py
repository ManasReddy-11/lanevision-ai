from __future__ import annotations

from pathlib import Path

import torch

from backend.app.ai.models.unet import LaneSegmentationUNet
from backend.training.losses import total_loss
from backend.training.utils import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_preserves_resume_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "resume_roundtrip.pth"

    model = LaneSegmentationUNet(in_channels=3, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    images = torch.randn(2, 3, 160, 320)
    masks = (torch.rand(2, 1, 160, 320) > 0.5).float()
    logits = model(images)
    loss = total_loss(logits, masks)
    loss.backward()
    optimizer.step()
    scheduler.step(0.1)

    save_checkpoint(
        model,
        checkpoint_path,
        epoch=3,
        metrics={"dice": 0.25, "iou": 0.15},
        optimizer=optimizer,
        scheduler=scheduler,
        best_metric=0.25,
        config={"device": "auto", "batch_size": 4},
    )

    restored_model = LaneSegmentationUNet(in_channels=3, num_classes=2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3, weight_decay=1e-4)
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer, mode="max", patience=2, factor=0.5)
    checkpoint = load_checkpoint(
        checkpoint_path,
        restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        map_location="cpu",
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_metric"] == 0.25
    assert checkpoint["config"]["device"] == "auto"
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
