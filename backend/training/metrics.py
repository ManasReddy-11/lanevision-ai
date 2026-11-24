from __future__ import annotations

import torch


def _to_binary_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return probs[:, 0:1, :, :]


def confusion_counts(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Pixel counts of the lane class at `threshold`; sum them over a split and call metrics_from_counts."""
    preds = (_to_binary_probabilities(logits.float()) >= threshold).float()
    targets = targets[:, 0:1, :, :].float()
    return {
        "tp": float((preds * targets).sum().item()),
        "fp": float((preds * (1.0 - targets)).sum().item()),
        "fn": float(((1.0 - preds) * targets).sum().item()),
        "tn": float(((1.0 - preds) * (1.0 - targets)).sum().item()),
    }


def metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts.get("tn", 0.0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "dice": float(2.0 * tp / (2.0 * tp + fp + fn + 1e-8)),
        "iou": float(tp / (tp + fp + fn + 1e-8)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / (precision + recall + 1e-8)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + 1e-8)),
    }


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Metrics of one batch. For a split, sum confusion_counts over the batches instead of averaging these."""
    return metrics_from_counts(confusion_counts(logits, targets))
