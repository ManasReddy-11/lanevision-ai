from __future__ import annotations

import torch


def _to_binary_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return probs[:, 0:1, :, :]


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = _to_binary_probabilities(logits)
    preds = (probs >= 0.5).float()
    targets = targets[:, 0:1, :, :].float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    union = tp + fp + fn
    iou = tp / (union + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }
