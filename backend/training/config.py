from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(slots=True)
class TrainingConfig:
    dataset_root: str = "dataset"
    image_extension: str = ".png"
    mask_extension: str = ".png"
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 0.001
    input_size: tuple[int, int] = (320, 160)
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    bce_weight: float = 0.7
    dice_weight: float = 0.3
    save_dir: str = "backend/models"
    production_checkpoint_path: str = "backend/models/lane_segmentation.pth"
    output_dir: str = "training_outputs"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        config = cls()
        training_settings = values.get("training", values)
        dataset_settings = values.get("dataset", {})
        if isinstance(training_settings, dict):
            for key, value in training_settings.items():
                if hasattr(config, key):
                    if key == "input_size" and isinstance(value, list):
                        setattr(config, key, tuple(value))
                    else:
                        setattr(config, key, value)
        if isinstance(dataset_settings, dict):
            for key, value in dataset_settings.items():
                if key == "root" and hasattr(config, "dataset_root"):
                    setattr(config, "dataset_root", value)
                elif key == "image_extension" and hasattr(config, "image_extension"):
                    setattr(config, "image_extension", value)
                elif key == "mask_extension" and hasattr(config, "mask_extension"):
                    setattr(config, "mask_extension", value)
        return config

    @classmethod
    def load(cls, path: str | None = None) -> "TrainingConfig":
        if path is None:
            return cls()
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "image_extension": self.image_extension,
            "mask_extension": self.mask_extension,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "input_size": list(self.input_size),
            "num_workers": self.num_workers,
            "seed": self.seed,
            "device": self.device,
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "save_dir": self.save_dir,
            "production_checkpoint_path": self.production_checkpoint_path,
            "output_dir": self.output_dir,
        }
