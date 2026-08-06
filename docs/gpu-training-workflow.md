# LaneVision AI GPU Training Workflow

This workflow uses the existing LaneVision AI U-Net, dataset converter, training loop, and evaluation path.
It is designed for Kaggle GPU sessions and does not require uploading the local 7.67 GiB prepared dataset.

## 1. Repository setup

Start from the repository root inside Kaggle working storage.

```bash
cd /kaggle/working/lanevision-ai
```

If the repo lives elsewhere inside `/kaggle/working`, adjust only the repository root path.

## 2. Preserve Kaggle CUDA PyTorch

Do not replace Kaggle's preinstalled CUDA-enabled PyTorch build with the local CPU build.
Install only the missing LaneVision AI dependencies.

Recommended pattern:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip install fastapi uvicorn[standard] pydantic pydantic-settings sqlalchemy python-multipart opencv-python-headless PyYAML httpx python-dotenv pytest
```

Only install `torch` or `torchvision` manually if the Kaggle runtime is missing them, and use a CUDA-compatible build if that happens.

## 3. Discover the raw TuSimple Kaggle dataset

The raw dataset is expected somewhere under `/kaggle/input`, but the exact mounted directory name should not be assumed.

Use the discovery helper:

```bash
python -m backend.training.kaggle_gpu_prep discover --search-root /kaggle/input
```

The helper inspects candidate roots and reports:
- annotation JSON files
- clip directories
- image directories
- sample `raw_file` values
- resolved image paths

## 4. Prepare the dataset into writable Kaggle storage

Do not modify `/kaggle/input`.
Prepare the dataset into `/kaggle/working`.

```bash
python -m backend.training.kaggle_gpu_prep prepare --search-root /kaggle/input --prepared-root /kaggle/working/tusimple_prepared
```

This uses the existing TuSimple converter and preserves local Windows behavior.

## 5. Run the CUDA preflight

```bash
python -m backend.training.kaggle_gpu_prep preflight --prepared-root /kaggle/working/tusimple_prepared --checkpoint-dir /kaggle/working/models --device auto
```

This fails clearly if CUDA is unavailable and reports:
- Python version
- torch version
- `torch.cuda.is_available()`
- CUDA version visible to PyTorch
- GPU device name
- GPU count
- selected training device
- dataset path
- train, val, and test counts
- checkpoint output directory
- free disk space

## 6. Run a small GPU smoke test

```bash
python -m backend.training.kaggle_gpu_prep smoke-test --prepared-root /kaggle/working/tusimple_prepared --checkpoint-dir /kaggle/working/models --config backend/training/configs/kaggle.yaml --device auto --batch-size 2 --train-batches 20 --val-batches 10
```

This uses real TuSimple samples on CUDA and reports:
- training and validation batch counts
- average train and validation loss
- validation Dice and IoU
- seconds per batch
- estimated minutes per epoch
- estimated time for 30 epochs
- peak GPU memory usage

## 7. Verify checkpoint path behavior

```bash
python -m backend.training.kaggle_gpu_prep verify-checkpoints --checkpoint-dir /kaggle/working/models --config backend/training/configs/kaggle.yaml
```

Expected writable artifacts:
- `/kaggle/working/models/lane_segmentation_best.pth`
- `/kaggle/working/models/lane_segmentation_last.pth`
- `/kaggle/working/models/lane_segmentation.pth`

Current persistence behavior:
- best checkpoint saves on validation improvement
- last checkpoint saves once per epoch
- resuming from `lane_segmentation_last.pth` is supported

## 8. Start or resume full training

Start:

```bash
python -m backend.training.train --config backend/training/configs/kaggle.yaml --dataset /kaggle/working/tusimple_prepared --device auto --batch-size 4 --epochs 30
```

Resume:

```bash
python -m backend.training.train --config backend/training/configs/kaggle.yaml --dataset /kaggle/working/tusimple_prepared --device auto --batch-size 4 --epochs 30 --resume /kaggle/working/models/lane_segmentation_last.pth
```

## 9. Evaluate the best checkpoint

```bash
python -m backend.training.evaluate --dataset /kaggle/working/tusimple_prepared/test --checkpoint /kaggle/working/models/lane_segmentation_best.pth --device auto --batch-size 4
```

## 10. Export the final checkpoint

Primary production artifact:

```text
/kaggle/working/models/lane_segmentation.pth
```

Additional training artifacts:

```text
/kaggle/working/models/lane_segmentation_best.pth
/kaggle/working/models/lane_segmentation_last.pth
/kaggle/working/training_outputs/
```

Download those files from Kaggle after training and evaluation complete.
