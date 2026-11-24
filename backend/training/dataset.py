from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import build_inference_transform


class LaneSegmentationDataset(Dataset):
    """Image/mask pairs from image_dir and mask_dir (matched by file name).

    preload=True decodes, resizes and stores every sample once (as uint8, in memory) using a thread pool, so epochs
    stop paying for PNG decoding of full-resolution frames. The tensors are identical to the unpreloaded ones; it needs
    roughly width * height * 4 bytes per sample (about 0.2 MB at 320x160, 0.8 MB at 640x320)."""

    def __init__(self, image_dir: str, mask_dir: str, input_size: tuple[int, int] = (320, 160), transform: Any | None = None,
                 preload: bool = False, preload_workers: int = 8) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.input_size = input_size
        self.transform = transform or build_inference_transform(input_size=input_size)
        self.image_paths = sorted(self.image_dir.glob("*"))
        self.mask_paths = sorted(self.mask_dir.glob("*"))
        self._validate()
        self._cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if preload:
            with ThreadPoolExecutor(max_workers=max(1, preload_workers)) as pool:
                self._cache = list(pool.map(self._load_compact, range(len(self.image_paths))))

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

    def _transformed(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Unable to read dataset sample {image_path.name}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image and mask dimensions do not match for {image_path.name}")
        return self.transform(image, mask)

    def _load_compact(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image, mask = self._transformed(index)
        # The transform's image is an exact multiple of 1/255 and its mask is {0, 1}: both fit losslessly in uint8.
        return (image * 255.0).round().to(torch.uint8).numpy(), mask.to(torch.uint8).numpy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is None:
            return self._transformed(index)
        image, mask = self._cache[index]
        return torch.from_numpy(image).float() / 255.0, torch.from_numpy(mask).float()
