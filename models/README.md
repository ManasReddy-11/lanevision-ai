# Model assets

This directory is intended for trained model checkpoints and other optional model assets.

## Expected checkpoint

Place a trained PyTorch checkpoint here or point the AI configuration to a different path:

- models/lane_segmentation.pth

## Configuration

Set the checkpoint path with:

```bash
AI_MODEL_PATH=models/lane_segmentation.pth
```

Large weights should not be committed directly to Git. Keep them local or store them in a release artifact or external storage location.
