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

## Notes

- The training pipeline is intentionally independent from the FastAPI routes.
- The current repository does not include a real lane dataset, so training will remain a framework until a real dataset is provided.
