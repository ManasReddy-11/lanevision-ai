from __future__ import annotations

import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs[:, 0:1, :, :]
        targets = targets[:, 0:1, :, :]
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)
        return 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth))


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits[:, 0:1, :, :], targets[:, 0:1, :, :])


def total_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.7, dice_weight: float = 0.3) -> torch.Tensor:
    return bce_weight * bce_loss(logits, targets) + dice_weight * DiceLoss()(logits, targets)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return DiceLoss()(logits, targets)
