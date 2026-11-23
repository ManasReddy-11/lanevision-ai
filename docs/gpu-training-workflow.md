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

Candidate roots are the search root, its immediate subdirectories, and every directory between the search root and an annotation file, so a dataset mounted several levels deep is found (for example `/kaggle/input/datasets/<owner>/<slug>/...`). Each candidate is scored on sampled records from every annotation file the converter would read under it.

A `raw_file` is resolved relative to its annotation file's directory, that directory's ancestors up to the source root, and the immediate subdirectories of those. This matches the official layout, where `test_label.json` sits next to `test_set/` and its `clips/...` paths are relative to `test_set/`. Only the full relative path counts: an image that merely shares the file name (every TuSimple keyframe is `20.jpg`) is never used.

Audited layout of the Kaggle dataset `manideep1108/tusimple` (version 5):

```text
<dataset root>/
  test_label_new.json            same 2782 records as TUSimple/test_label.json, raw_file spelled test_set/clips/...
  TUSimple/
    test_label.json              2782 test records, raw_file clips/... relative to test_set/
    test_set/clips/...           test frames
    test_set/test_tasks_0627.json  test task list without lane points
    train_set/label_data_0313.json, label_data_0531.json, label_data_0601.json   2858 + 358 + 410 records
    train_set/clips/...          training frames
    train_set/seg_label/...      segmentation PNGs plus test.json/train_val.json copies (not read by the converter)
```

## 4. Prepare the dataset into writable Kaggle storage

Do not modify `/kaggle/input`.
Prepare the dataset into `/kaggle/working`.

```bash
python -m backend.training.kaggle_gpu_prep prepare --search-root /kaggle/input --prepared-root /kaggle/working/tusimple_prepared --split-mode official
```

This uses the existing TuSimple converter and preserves local Windows behavior.

If discovery cannot pick the right directory, pass the TuSimple root explicitly with `--raw-root <directory that holds the annotation JSON files and clips>`.

Split modes:
- `--split-mode official` (recommended for TuSimple): train = `label_data_0313.json` + `label_data_0601.json`, val = `label_data_0531.json`, test = `test_label*.json`. The official test images never enter training, so test metrics stay comparable with the TuSimple benchmark. Files outside these splits (such as `test_tasks_0627.json`) are listed in the manifest as `ignored_annotation_files`. Conversion stops if one image is annotated in two splits or if a split has no annotation file.
- `--split-mode random` (default, previous behavior): pools every annotation file, including the official test labels, and splits the unique samples at random (`--split-seed`, 10% val, 10% test).

Split and integrity guarantees of the converter:
- every valid sample is written to exactly one of `train/`, `val/`, or `test/`; validation and test samples are moved out of `train/`, so the splits never share an image
- an image referenced by several annotation files is converted once. Samples are identified by their path from `clips/` onward, so `test_label.json` (`clips/...`) and `test_label_new.json` (`test_set/clips/...`) refer to the same samples. The first valid reference in sorted file order wins, and the manifest reports `duplicate_references` and `conflicting_duplicates` (same image, different lanes)
- if two different samples resolve to the same image file or output name, only the first is kept and the manifest reports `ambiguous_samples`; a non-zero value means image resolution needs review before training
- conversion refuses an output root whose split folders already contain files; prepare into an empty directory
- `prepare` refuses to reuse an existing prepared dataset whose splits overlap

Datasets prepared before these guarantees were added kept every validation and test sample in `train/` as well. Delete such a dataset and prepare it again; do not train or report metrics on it.

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
