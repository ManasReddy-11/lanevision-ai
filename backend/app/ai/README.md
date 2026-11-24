# AI subsystem

This package contains the neural segmentation path for LaneVision AI.

## Files

- adapter.py: runs the model and turns its output into lanes, the ego lane and the vehicle offset (`LaneDetectionResult`)
- lane_geometry.py: multi-lane extraction from the lane-probability map, polynomial fitting, ego-lane selection
- registry.py: one shared detector (one loaded model) per configuration per process
- config.py: environment-driven configuration for the AI mode
- inference.py: preprocessing plus a forward pass, returning the lane probability map
- model_loader.py: thread-safe, load-once checkpoint loading and compatibility handling
- preprocessing.py: deterministic image preprocessing
- postprocessing.py: mask thresholding, resizing, and noise filtering (for the overlay)
- overlay.py: mask tint, every detected lane (yellow), ego-lane boundaries (green) and ego-lane area
- models/unet.py: lightweight U-Net implementation

## Model checkpoint

A real trained checkpoint is required for lane segmentation inference. Point the application at it with:

```bash
AI_MODEL_PATH=/path/to/lane_segmentation.pth   # AI_DEVICE=auto|cpu|cuda (default auto)
```

Without a usable checkpoint, image and video requests with `algorithm=ai` fail with HTTP 503 and a `detail` such as `AI model unavailable: AI model weights were not found (lane_segmentation.pth). Set AI_MODEL_PATH to a trained checkpoint file ...` (a broken file gives `AI model unavailable: Unable to load AI checkpoint: ...`). Nothing is stored, no frame is processed, the failure is recorded in the history, and the classic detector is never substituted. A missing or broken checkpoint is detected once per process and reported on every request. (Called directly, `AILaneDetectorAdapter.detect` still returns the unmodified frame with the warning `AI model unavailable: ...`.)

Expected format:
- PyTorch state dict or checkpoint dictionary containing a model state dict
- The loader expects the architecture defined in models/unet.py and will reject incompatible checkpoints with a clear error

Output contract (must match training):
- The U-Net returns two output channels, but training (`backend/training/losses.py` and `metrics.py`) optimizes and scores only channel 0, with a sigmoid
- Inference therefore reads the lane probability as `sigmoid(logits[AI_LANE_CLASS_ID])` with `AI_LANE_CLASS_ID=0` by default; channel 1 is never trained and must not be used
- The probability map is resized back to the input frame's height and width, so non-square frames such as 1280x720 keep their shape

## Lanes, ego lane and offset

- Every visible lane is extracted (not just two): rows are scanned bottom-up, above-threshold runs are linked into tracks by slope prediction (bridging dashes and short occlusions), fragments are merged, and each lane is fitted with x = f(y)
- The ego lane is formed by the lanes immediately left and right of the camera reference point (the frame centre by default) at a reference row at 90% of the frame height
- `vehicle_offset_pixels = camera_x - ego_lane_centre_x`: positive means the vehicle is right of the lane centre, as in `cv/metrics/lane_metrics.py`. It is in pixels at the reference row; no real-world calibration is applied. `vehicle_offset_normalized` divides it by the ego-lane width
- The offset, `lane_center_x` and `lane_width_bottom_pixels` are `null`, with the warning `Vehicle offset unavailable: <reason>`, when a boundary is missing, only seen far from the vehicle, low-confidence, or the lane width is implausible
- `confidence` is the mean, over the two ego boundaries, of each boundary's mean lane probability scaled by its vertical coverage (0 when no boundary was found)
- `detected_segments` in the API response is the number of lanes detected
- The lane probability threshold (0.3) and dash-bridging gap (12 rows) were selected on the TuSimple validation split, not the test split

Runtime behavior:
- The checkpoint is read once per process: `get_detector("ai")` returns a shared detector, and a video reuses one detector for all of its frames
- Inference uses `model.eval()` and `torch.inference_mode()`
- CPU inference is supported by default
- CUDA is used automatically when `AI_DEVICE=cuda` and a GPU is available
- If CUDA is requested but unavailable, the loader falls back to CPU cleanly
- Image and video responses report the same `ai_metadata`: the model's actual device, whether its checkpoint is loaded, the checkpoint file name, input size, and any load error

Important:
- This repository does not ship a trained checkpoint; weights are excluded from Git
