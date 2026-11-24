from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import yaml

SCHEDULERS = ("plateau", "cosine", "step", "none")
LOSSES = ("bce_dice", "focal_dice", "tversky", "focal_tversky", "bce")
MONITORED_METRICS = ("dice", "iou", "recall", "f1", "tusimple_accuracy")


@dataclass(slots=True)
class TrainingConfig:
    # Data
    dataset_root: str = "dataset"
    image_extension: str = ".png"
    mask_extension: str = ".png"
    input_size: tuple[int, int] = (320, 160)  # model input (width, height); saved in checkpoints, used by inference
    train_mask_dilation: int = 0  # thicken TRAIN targets by this many pixels each side (at prepared-mask resolution);
    #                               val/test masks are never changed, so pixel metrics keep one definition
    num_workers: int = 0
    preload: bool = False  # decode and resize every train/val sample once, into memory (much faster epochs)
    tusimple_val_labels: list[str] | None = None  # TuSimple label files of the val images: adds per-epoch official
    #                                               accuracy / FP / FN on the validation split (monitor: tusimple_accuracy)

    # Optimisation
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = False  # automatic mixed precision (float16 autocast + gradient scaling), CUDA only

    # Learning-rate schedule
    scheduler: str = "plateau"  # plateau | cosine | step | none
    scheduler_patience: int | None = None  # plateau: epochs without improvement before decaying (None: max(1, patience // 2))
    scheduler_factor: float = 0.5  # plateau and step: decay factor
    step_size: int = 10  # step: epochs between decays
    warmup_epochs: int = 0  # linear warm-up from learning_rate / 10 (cosine and step)
    min_learning_rate: float = 1e-6  # cosine: final learning rate

    # Loss
    loss: str = "bce_dice"  # bce_dice | focal_dice | tversky | focal_tversky | bce
    bce_weight: float = 0.7
    dice_weight: float = 0.3
    pos_weight: float | None = None  # BCE weight of lane pixels (None: unweighted)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75  # focal: weight of lane pixels (background gets 1 - alpha)
    tversky_alpha: float = 0.3  # Tversky: weight of false positives
    tversky_beta: float = 0.7  # Tversky: weight of false negatives (> alpha favours recall: fewer missed lane pixels)

    # Model selection: validation split only
    monitor: str = "dice"  # validation metric that selects the best epoch (maximised)
    early_stopping: bool = True
    patience: int = 5  # epochs without an improvement of at least min_delta before stopping
    min_delta: float = 0.001
    restore_best: bool = True  # load the best epoch's weights at the end (and for the final reports)
    evaluate_test: bool = False  # report test metrics of the best model after training (never used for selection)

    # Reproducibility
    seed: int = 42
    deterministic: bool = False  # deterministic cuDNN kernels (slower); seeds are always set
    device: str = "auto"

    # Outputs
    experiment_name: str = "experiment"
    save_dir: str = "backend/models"  # checkpoints: lane_segmentation_best.pth and lane_segmentation_last.pth
    output_dir: str = "training_outputs"  # config, metrics.jsonl, training_history.json, summary.json, curves.png
    production_checkpoint_path: str | None = None  # export the restored best model here; None: no export. An
    #                                                existing file is never overwritten unless overwrite_production

    overwrite_production: bool = False

    def validate(self) -> "TrainingConfig":
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {self.loss!r}")
        if self.monitor not in MONITORED_METRICS:
            raise ValueError(f"monitor must be one of {MONITORED_METRICS}, got {self.monitor!r}")
        if len(self.input_size) != 2 or min(self.input_size) <= 0 or any(side % 16 for side in self.input_size):
            raise ValueError(f"input_size must be (width, height), both multiples of 16 (four U-Net poolings); got {self.input_size}")
        if self.train_mask_dilation < 0:
            raise ValueError("train_mask_dilation must be >= 0")
        if self.monitor == "tusimple_accuracy" and not self.tusimple_val_labels:
            raise ValueError("monitor tusimple_accuracy needs tusimple_val_labels")
        if isinstance(self.tusimple_val_labels, str):
            self.tusimple_val_labels = [self.tusimple_val_labels]
        return self

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        config = cls()
        names = {field.name for field in fields(cls)}
        training_settings = values.get("training", values)
        dataset_settings = values.get("dataset", {})
        if isinstance(training_settings, dict):
            for key, value in training_settings.items():
                if key in names:
                    setattr(config, key, tuple(value) if key == "input_size" and isinstance(value, list) else value)
        if isinstance(dataset_settings, dict):
            for key, value in dataset_settings.items():
                if key == "root":
                    config.dataset_root = value
                elif key in ("image_extension", "mask_extension", "train_mask_dilation"):
                    setattr(config, key, value)
        return config  # validated after command-line overrides (train() validates too)

    @classmethod
    def load(cls, path: str | None = None) -> "TrainingConfig":
        if path is None:
            return cls()
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["input_size"] = list(self.input_size)
        return values
