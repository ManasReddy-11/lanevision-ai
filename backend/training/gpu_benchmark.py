"""Measure training-step throughput, peak GPU memory and single-frame inference latency of the U-Net.

Synthetic inputs only (no dataset, no training run): it answers "how fast and how much memory" for each input size,
batch size and precision before an experiment is scheduled. Data loading is not included; compare with an epoch's
train_images_per_second in metrics.jsonl to see how much of a real epoch is spent loading data.

    python -m backend.training.gpu_benchmark --sizes 320x160 640x320 --batch-sizes 4 8 16 --amp off on --out bench.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any

import torch

from backend.app.ai.models.unet import LaneSegmentationUNet
from backend.training.config import TrainingConfig
from backend.training.losses import build_loss


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_step_benchmark(size: tuple[int, int], batch_size: int, amp: bool, device: torch.device, steps: int, warmup: int) -> dict[str, Any]:
    width, height = size
    model = LaneSegmentationUNet(in_channels=3, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = build_loss(TrainingConfig())
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    images = torch.rand(batch_size, 3, height, width, device=device)
    masks = (torch.rand(batch_size, 1, height, width, device=device) > 0.98).float()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(warmup + steps):
            if step == warmup:
                _sync(device)
                started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(images)
            loss = loss_fn(logits.float(), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        _sync(device)
        seconds = time.perf_counter() - started
    except torch.cuda.OutOfMemoryError:
        return {"input_size": [width, height], "batch_size": batch_size, "amp": amp, "status": "out of memory"}
    result: dict[str, Any] = {"input_size": [width, height], "batch_size": batch_size, "amp": amp, "status": "ok",
                              "step_ms": round(1000 * seconds / steps, 2), "train_images_per_second": round(batch_size * steps / seconds, 1)}
    if device.type == "cuda":
        result["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / 2**20, 1)
    del model, optimizer, images, masks
    return result


def inference_benchmark(size: tuple[int, int], amp: bool, device: torch.device, frames: int, warmup: int) -> dict[str, Any]:
    width, height = size
    model = LaneSegmentationUNet(in_channels=3, num_classes=2).to(device).eval()
    image = torch.rand(1, 3, height, width, device=device)
    timings = []
    with torch.inference_mode():
        for index in range(warmup + frames):
            _sync(device)
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                model(image)
            _sync(device)
            if index >= warmup:
                timings.append(1000 * (time.perf_counter() - started))
    timings.sort()
    return {"input_size": [width, height], "amp": amp, "model_ms_mean": round(sum(timings) / len(timings), 2), "model_ms_p95": round(timings[int(0.95 * (len(timings) - 1))], 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", default=["320x160", "640x320"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--amp", nargs="+", choices=["off", "on"], default=["off", "on"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--inference-frames", type=int, default=50)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    torch.backends.cudnn.benchmark = True
    sizes = [tuple(int(v) for v in size.lower().split("x")) for size in args.sizes]
    amps = [mode == "on" for mode in args.amp] if device.type == "cuda" else [False]
    report: dict[str, Any] = {"device": str(device), "torch": torch.__version__, "platform": platform.platform(), "threads": torch.get_num_threads(),
                              "parameters": sum(p.numel() for p in LaneSegmentationUNet(in_channels=3, num_classes=2).parameters()), "train": [], "inference": []}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update(gpu=properties.name, gpu_memory_mb=round(properties.total_memory / 2**20), cuda=torch.version.cuda)
    for size in sizes:
        for amp in amps:
            report["inference"].append(inference_benchmark(size, amp, device, args.inference_frames, args.warmup))
            print(json.dumps(report["inference"][-1]), flush=True)
            if args.skip_training:
                continue
            for batch_size in args.batch_sizes:
                report["train"].append(train_step_benchmark(size, batch_size, amp, device, args.steps, args.warmup))
                print(json.dumps(report["train"][-1]), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
