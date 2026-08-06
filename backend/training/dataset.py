from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import build_inference_transform


class LaneSegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, input_size: tuple[int, int] = (320, 160), transform: Any | None = None) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.input_size = input_size
        self.transform = transform or build_inference_transform(input_size=input_size)
        self.image_paths = sorted(self.image_dir.glob("*"))
        self.mask_paths = sorted(self.mask_dir.glob("*"))
        self._validate()

    def _validate(self) -> None:
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise ValueError("Dataset directories must exist")
        if not self.image_paths:
            raise ValueError("No images found in dataset")
        if not self.mask_paths:
            raise ValueError("No masks found in dataset")
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Image and mask counts must match")

        for image_path in self.image_paths:
            if image_path.is_dir():
                continue
            mask_path = self.mask_dir / image_path.name
            if not mask_path.exists():
                raise ValueError(f"Image/mask pair mismatch for {image_path.name}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Unable to read dataset sample {image_path.name}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image and mask dimensions do not match for {image_path.name}")
        return self.transform(image, mask)
