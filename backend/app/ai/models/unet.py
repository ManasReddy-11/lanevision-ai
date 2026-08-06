from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LaneSegmentationUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(256, 128, kernel_size=1, bias=False))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(128, 64, kernel_size=1, bias=False))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(64, 32, kernel_size=1, bias=False))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(32, 32, kernel_size=1, bias=False))
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

        self.dec4 = DoubleConv(256, 128)
        self.dec3 = DoubleConv(128, 64)
        self.dec2 = DoubleConv(64, 32)
        self.dec1 = DoubleConv(32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        dec4 = self.dec4(torch.cat([self.up4(enc4), enc3], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc2], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc1], dim=1))
        dec1 = self.dec1(self.up1(dec2))
        dec1 = F.interpolate(dec1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(dec1)
