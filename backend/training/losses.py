"""Segmentation losses on the lane channel (sigmoid of channel 0, as inference reads it).

bce_dice (the baseline) = bce_weight * BCE + dice_weight * Dice. The alternatives target the thin-lane imbalance (lane
pixels are about 1.3% of a TuSimple mask): focal BCE down-weights easy background pixels, and Tversky weights false
negatives (missed lane pixels) more than false positives when beta > alpha.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def _lane(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return logits[:, 0:1, :, :].float(), targets[:, 0:1, :, :].float()


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _lane(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)
        return 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth))


def bce_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    logits, targets = _lane(logits, targets)
    weight = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    logits, targets = _lane(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)  # probability given to the true class
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return torch.mean(alpha_t * (1.0 - p_t) ** gamma * bce)


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.0, smooth: float = 1e-6) -> torch.Tensor:
    """1 - Tversky index (alpha weights false positives, beta false negatives); gamma < 1 gives the focal Tversky loss."""
    logits, targets = _lane(logits, targets)
    probs = torch.sigmoid(logits)
    tp = torch.sum(probs * targets)
    fp = torch.sum(probs * (1.0 - targets))
    fn = torch.sum((1.0 - probs) * targets)
    index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1.0 - index) ** gamma


def total_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.7, dice_weight: float = 0.3) -> torch.Tensor:
    return bce_weight * bce_loss(logits, targets) + dice_weight * DiceLoss()(logits, targets)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return DiceLoss()(logits, targets)


def build_loss(config) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """The training loss selected by a TrainingConfig (config.loss and its parameters)."""
    dice = DiceLoss()
    if config.loss == "bce_dice":
        return lambda logits, targets: config.bce_weight * bce_loss(logits, targets, config.pos_weight) + config.dice_weight * dice(logits, targets)
    if config.loss == "bce":
        return lambda logits, targets: bce_loss(logits, targets, config.pos_weight)
    if config.loss == "focal_dice":
        return lambda logits, targets: config.bce_weight * focal_loss(logits, targets, config.focal_gamma, config.focal_alpha) + config.dice_weight * dice(logits, targets)
    if config.loss == "tversky":
        return lambda logits, targets: tversky_loss(logits, targets, config.tversky_alpha, config.tversky_beta)
    if config.loss == "focal_tversky":
        return lambda logits, targets: tversky_loss(logits, targets, config.tversky_alpha, config.tversky_beta, gamma=0.75)
    raise ValueError(f"Unknown loss: {config.loss}")
