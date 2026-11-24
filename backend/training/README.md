# LaneVision AI training pipeline

This package provides an independent training workflow for the U-Net segmentation model used by the AI inference path.

## Structure

- train.py: training entrypoint (early stopping, best-epoch restore, resume, schedules, AMP, structured run logs)
- config.py: `TrainingConfig`, every training option; configs/*.yaml set them, `--set KEY=VALUE` overrides any of them
- dataset.py, transforms.py: image/mask pairs resized to the model input; optional in-memory preload; train-only mask dilation
- losses.py: bce_dice (default), focal_dice, tversky, focal_tversky, bce
- metrics.py: pixel Dice, IoU, precision, recall, F1 from counts summed over a split
- evaluate.py: pixel metrics of a checkpoint on one split
- tusimple_benchmark.py: official TuSimple accuracy / FP / FN (see below)
- tusimple_temporal_eval.py: video (20-frame clip) evaluation of lane tracking
- gpu_benchmark.py: training-step throughput, peak GPU memory and inference latency on synthetic inputs

## Dataset layout

```text
dataset/
  train/images, train/masks
  val/images,   val/masks
  test/images,  test/masks
```

Each image has a mask with the same file name and size. `backend/training/datasets/tusimple.py` prepares this layout from
TuSimple (official split: train = label_data_0313 + 0601, val = 0531, test = test_label.json, one copy of each distinct
picture). Training refuses to start if a file name appears in two splits (`--check-content` also compares image bytes).

## Training

```bash
python -m backend.training.train --config backend/training/configs/default.yaml --dataset path/to/dataset \
    --set experiment_name=my_run --set epochs=30 --set preload=true
```

- **Model selection uses the validation split only.** The best epoch maximises `monitor` (validation Dice over the whole
  split by default). With `tusimple_val_labels` (the label file of the val images, e.g. `train_set/label_data_0531.json`)
  every epoch also reports the official TuSimple accuracy / FP / FN of the val split, and `monitor: tusimple_accuracy`
  selects on it. The test split is read only after training, with the selected weights, when `evaluate_test: true`.
- **Early stopping**: stops after `patience` epochs without an improvement of at least `min_delta`; with `restore_best`
  the best epoch's weights are reloaded and re-validated at the end.
- **Reproducibility**: `seed` seeds Python, NumPy, PyTorch and DataLoader workers, and each epoch's shuffling;
  `deterministic: true` also selects deterministic cuDNN kernels.
- **Resume**: `--resume <save_dir>/lane_segmentation_last.pth` continues a run with its optimizer, scheduler, AMP scaler,
  early-stopping and RNG state (a resumed run reproduces the uninterrupted one).
- **Schedules**: `scheduler: plateau | cosine | step | none`, with `warmup_epochs` for cosine and step.
- **Mixed precision**: `amp: true` (CUDA only; ignored on CPU).
- **Input resolution**: `input_size: [width, height]` (multiples of 16). It is saved in the checkpoint; the app's model
  loader and evaluate.py serve and score the checkpoint at that size.
- **Thicker training targets**: `train_mask_dilation: N` widens train masks by N pixels on each side (at the prepared
  1280x720 resolution) before resizing. Validation and test masks are never changed, so pixel metrics keep one definition.
- **Losses**: `loss` and its parameters (`bce_weight`, `dice_weight`, `pos_weight`, `focal_gamma`, `focal_alpha`,
  `tversky_alpha`, `tversky_beta`).
- **Faster epochs**: `preload: true` decodes and resizes every sample once into memory (about 0.2 MB per sample at
  320x160, 0.8 MB at 640x320); `num_workers` loads batches in parallel otherwise.

Outputs (`output_dir`): `config.json` (configuration, git commit, dataset counts and file-list fingerprints, environment),
`metrics.jsonl` and `training_history.json` (per epoch: losses, metrics, learning rate, epoch time, throughput, peak GPU
memory), `summary.json` (best epoch, stop reason, restored-best metrics, optional test metrics) and `curves.png` (needs
matplotlib). Checkpoints (`save_dir`): `lane_segmentation_best.pth` and `lane_segmentation_last.pth`.

**The production checkpoint is protected.** Training exports a model for the app only when `production_checkpoint_path`
is set, writes the best epoch's weights without optimizer or RNG state, and never overwrites an existing file unless
`overwrite_production: true`. Point `AI_MODEL_PATH` at a new model to try it; keep the previous file.

## Evaluation

```bash
python -m backend.training.evaluate --dataset path/to/dataset/val --checkpoint path/to/lane_segmentation.pth
```

Pixel metrics at threshold 0.5, from counts summed over the split (the same definition training uses).

## TuSimple benchmark

`tusimple_benchmark.py` scores a checkpoint with the official TuSimple metrics (accuracy, FP, FN; a port of the
benchmark's `LaneEval`) and the training pixel metrics (Dice, IoU, precision, recall at threshold 0.5). Predictions come
from the production inference path: the model's probability map, multi-lane extraction (`app/ai/lane_geometry.py`), and
sampling at each record's `h_samples`. Run it from the repository root:

```bash
# Official, unmodified test split (all 2782 records of test_label.json)
python -m backend.training.tusimple_benchmark run --checkpoint path/to/lane_segmentation.pth \
    --labels <dataset root>/TUSimple/test_label.json --source-root <dataset root> \
    --out-dir eval/test_official --device cpu --extend-bottom

# Deduplicated test split: the same per-image results restricted to the prepared test images
python -m backend.training.tusimple_benchmark summarize --per-image eval/test_official/per_image.jsonl \
    --only-names <prepared root>/test/images --out eval/test_dedup.json
```

Outputs: `results.json` (metrics, run-time statistics, checkpoint hash and epoch, settings), `predictions.jsonl` (TuSimple
submission format), `per_image.jsonl`, and worst/median/best example images (ground-truth points red, predictions green).

- Tune post-processing settings (`--geometry`, `--extend-bottom`) on the validation split
  (`train_set/label_data_0531.json` with `--only-names <prepared root>/val/images`), never on the test split.
- The official rule scores an image 0 when its run time exceeds 200 ms. Run time is measured on the evaluating machine,
  so run the benchmark without other heavy work on the same CPU or GPU and check `images_over_run_time_limit`.
- The official test split counts 12 pictures twice (stored as both `clips/0530/<id>_0/20.jpg` and
  `clips/0531/<id>/20.jpg`); the deduplicated split keeps one copy. Seven test pictures also appear in
  `label_data_0531.json`; the deduplicated preparation drops them from validation, so they were not used for checkpoint
  selection.

## Notes

- The training pipeline is independent from the FastAPI routes; datasets and checkpoints are never committed.
- Validation metrics in runs before this version were averaged per batch; they are now computed over the whole split,
  as evaluate.py and the TuSimple benchmark do, so the numbers are not directly comparable with older training logs.
