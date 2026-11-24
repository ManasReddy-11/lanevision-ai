# LaneVision AI training pipeline

This package provides an independent training workflow for the U-Net segmentation model used by the AI inference path.

## Structure

- train.py: end-to-end training entrypoint
- dataset.py: semantic segmentation dataset loader
- transforms.py: preprocessing and optional augmentation
- losses.py: BCE + Dice loss
- metrics.py: Dice, IoU, precision, recall, and F1
- evaluate.py: standalone evaluation CLI
- config.py: dataclass-based training config
- configs/default.yaml: default training settings

## Dataset layout

Expected dataset structure:

```text
dataset/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Each image must have a matching mask of the same dimensions.

## Training command

```bash
python -m training.train --dataset path/to/dataset --epochs 30 --batch-size 8 --device auto
```

## Evaluation command

```bash
python -m training.evaluate --dataset path/to/test --checkpoint backend/models/lane_segmentation.pth
```

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

- The training pipeline is intentionally independent from the FastAPI routes.
- The current repository does not include a real lane dataset, so training will remain a framework until a real dataset is provided.
