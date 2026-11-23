# AI subsystem

This package contains the neural segmentation path for LaneVision AI.

## Files

- adapter.py: bridges the neural segmentation output into the existing LaneDetectionResult contract
- config.py: environment-driven configuration for the AI mode
- inference.py: cached inference engine using PyTorch
- model_loader.py: checkpoint loading and compatibility handling
- preprocessing.py: deterministic image preprocessing
- postprocessing.py: mask thresholding, resizing, and noise filtering
- models/unet.py: lightweight U-Net implementation

## Model checkpoint

A real trained checkpoint is required for actual lane segmentation inference. Without one, the AI mode reports a clear warning and returns a graceful fallback response.

Expected checkpoint location:
- backend/models/lane_segmentation.pth

Expected format:
- PyTorch state dict or checkpoint dictionary containing a model state dict
- The loader expects the architecture defined in models/unet.py and will reject incompatible checkpoints with a clear error

Output contract (must match training):
- The U-Net returns two output channels, but training (`backend/training/losses.py` and `metrics.py`) optimizes and scores only channel 0, with a sigmoid
- Inference therefore reads the lane probability as `sigmoid(logits[AI_LANE_CLASS_ID])` with `AI_LANE_CLASS_ID=0` by default; channel 1 is never trained and must not be used
- The post-processed mask is resized back to the input frame's height and width, so non-square frames such as 1280x720 keep their shape

Runtime behavior:
- The model is loaded once and cached for reuse
- Inference uses `model.eval()` and `torch.inference_mode()`
- CPU inference is supported by default
- CUDA is used automatically when `AI_DEVICE=cuda` and a GPU is available
- If CUDA is requested but unavailable, the loader falls back to CPU cleanly

Important:
- This repository does not ship a trained checkpoint
- The current AI path is architecture-ready, but real neural inference is not verified until a real checkpoint is provided
