from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..core.config import get_settings


@dataclass(slots=True)
class AIConfig:
    enabled: bool = True
    model_path: str | None = None
    device: str = "auto"
    input_width: int = 320
    input_height: int = 160
    mask_threshold: float = 0.5
    min_component_area: int = 64
    overlay_alpha: float = 0.35
    warmup: bool = True
    model_name: str = "lane_segmentation_unet"
    architecture: str = "u-net"
    lane_class_id: int = 1
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_settings(cls) -> "AIConfig":
        settings = get_settings()
        return cls(
            enabled=settings.ai_enabled,
            model_path=settings.ai_model_path or None,
            device=settings.ai_device,
            input_width=settings.ai_input_width,
            input_height=settings.ai_input_height,
            mask_threshold=settings.ai_mask_threshold,
            min_component_area=settings.ai_min_component_area,
            overlay_alpha=settings.ai_overlay_alpha,
            warmup=settings.ai_warmup,
            model_name=settings.ai_model_name,
            architecture=settings.ai_architecture,
            lane_class_id=settings.ai_lane_class_id,
        )
