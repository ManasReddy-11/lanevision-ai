from __future__ import annotations

import cv2
import numpy as np
import torch


class LaneSegmentationTransform:
    """Resize an image/mask pair to the model input and convert to tensors (image RGB in [0, 1], mask {0, 1}).

    mask_dilation thickens the lane target by that many pixels on each side, at the mask's own (prepared) resolution and
    before resizing. It is meant for training targets only: evaluation masks must keep one fixed definition."""

    def __init__(self, input_size: tuple[int, int], augment: bool = False, mask_dilation: int = 0) -> None:
        self.input_size = tuple(input_size)
        self.augment = augment
        self.mask_dilation = int(mask_dilation)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if image is None or mask is None:
            raise ValueError("Image and mask are required")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a 3-channel BGR image")
        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D array")

        if self.mask_dilation > 0:
            size = 2 * self.mask_dilation + 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask.astype(np.float32), self.input_size, interpolation=cv2.INTER_NEAREST)
        resized_mask = (resized_mask >= 0.5).astype(np.float32)

        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1) / 255.0
        tensor_mask = torch.from_numpy(resized_mask).float().unsqueeze(0)
        return tensor_image, tensor_mask


def build_train_transforms(input_size: tuple[int, int], augment: bool = False, mask_dilation: int = 0) -> LaneSegmentationTransform:
    return LaneSegmentationTransform(input_size=input_size, augment=augment, mask_dilation=mask_dilation)


def build_inference_transform(input_size: tuple[int, int]) -> LaneSegmentationTransform:
    return LaneSegmentationTransform(input_size=input_size, augment=False)
