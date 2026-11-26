# Model optimisation plan (proposal, awaiting approval)

Nothing in this plan has been run. It proposes controlled GPU experiments to improve the U-Net lane segmentation model,
with the cost of each. Numbers are labelled **measured** (with their source) or **estimate**.

## Starting point (measured)

Baseline checkpoint: TuSimple U-Net (1.82 M parameters), 320x160 input, epoch 10 of a 30-epoch run on a Kaggle T4.

| | Accuracy | FP | FN | Pixel Dice | Recall | Precision |
|---|---|---|---|---|---|---|
| Official test split (2,782 images, single frame) | 0.8697 | 0.1031 | 0.1970 | 0.5234 | 0.6105 | 0.4581 |
| Validation split (351 images, single frame) | 0.8169 | 0.1569 | 0.2645 | | | |
| Validation clips with video tracking (frame 20) | 0.8862 | 0.1571 | 0.1717 | | | |

What limits it (measured):

- **Missed lanes dominate.** FN (0.197) is about twice FP (0.103) on the test split; images have 3.58 ground-truth
  lanes on average and the model reports 3.04. Missed lanes are mostly faint or far outer lanes (worst-case examples
  of the benchmark).
- **Targets are 1-2 px wide at the model resolution.** The prepared masks draw lanes 5-7 px wide at 1280x720; nearest-
  neighbour resizing to 320x160 leaves 1-2 px, so far and faint lanes are thin, aliased targets.
- **Overfitting after epoch 10.** Validation Dice peaked at epoch 10 (0.516) while training loss kept falling
  (0.185 -> 0.133 by epoch 30) and validation recall fell (0.60 -> 0.49). The old training loop had no early stopping.
  There is no data augmentation.
- **Slow epochs.** 320 s per epoch on a T4 at 320x160, batch 4, float32, `num_workers: 0` (stage-2 log), about 10
  training images per second. The GPU's training compute alone is 24.3 images per second at that setting (134 s per
  epoch, synthetic benchmark below), so roughly 186 s per epoch went to decoding PNGs, resizing and validation; decoding
  a 1280x720 image and mask PNG and resizing takes about 25 ms per sample on a desktop CPU (single thread, measured).

## Common protocol for every experiment

- **Data:** the official split as prepared before (train = `label_data_0313` + `0601`, 3,268 images; val = `0531`, 351;
  test = `test_label.json`, 2,770 deduplicated / 2,782 official). Same preparation code and checks; split overlap is
  rejected by the training script.
- **Code:** a fixed commit of `feature/ai-lane-segmentation`; configs in `backend/training/configs/experiments/`. B to D
  each change exactly one factor relative to A (enforced by a test).
- **Model selection on validation only:** `monitor: tusimple_accuracy` (official accuracy of the 351 validation images,
  production lane extraction), early stopping with patience 6 and min_delta 0.002, at most 30 epochs, best epoch restored.
  Pixel metrics are reported on the unchanged validation masks.
- **Seeds:** 42 for every experiment; A is also run with seed 7. A difference between experiments smaller than the
  difference between A's two seeds is not treated as an improvement.
- **The test split is not used** until a final model is chosen (step E). It is then evaluated once, single-frame and
  with video tracking, and compared with the baseline under the same post-processing.
- **Post-processing:** the lane-extraction threshold and gap (0.3, 12 rows) were tuned for the baseline on validation.
  The final candidate gets the same small validation grid (threshold 0.3 / 0.4 / 0.5, gap 6 / 12) before the test run.
- **Speed-ups:** `preload: true` (samples decoded once; identical tensors, tested) and two loader workers, which do not
  change results; and mixed precision (`amp: true`) in every experiment, measured 2.4x faster at 320x160 and needed for
  batch 16 at 640x320. All of A-E use it, so comparisons stay controlled; A also checks that it reproduces the
  float32 baseline.
- **Outputs:** each run writes `config.json`, `metrics.jsonl`, `summary.json`, `curves.png` and its checkpoints under
  `/kaggle/working/runs/<experiment>/`, downloaded afterwards to `D:\lanevision-data\experiments\<experiment>\`.
  Nothing is written to the repository or to the production checkpoint (`production_checkpoint_path: null`); the
  baseline checkpoint and its read-only copy stay as they are.
- **Recorded per experiment:** validation accuracy / FP / FN, validation pixel Dice / IoU / precision / recall, best
  epoch, epochs run, wall time per epoch, peak GPU memory, and (for B) CPU inference latency at the new resolution.

### Early termination (all experiments)

- Loss becomes NaN or infinite: stop.
- Out of GPU memory: stop; rerun with half the batch size and say so in the results.
- An epoch takes more than twice the estimate below after the second epoch: stop and investigate the data pipeline.
- Validation accuracy at epoch 8 more than 0.05 below A's at epoch 8 (same seed): stop as clearly worse.
- Early stopping (patience 6) ends every run normally.

## GPU measurements for the estimates

**Measured** on a Kaggle Tesla T4 (14.9 GB usable), torch 2.11 / CUDA 13.0, `backend/training/gpu_benchmark.py` at
commit eb7cd22: random inputs, U-Net forward, BCE + Dice loss, backward and AdamW step; 40 timed steps after 10 warm-up
steps. Data loading and validation are not included.

| Input | Batch | Precision | Training images / s | Peak GPU memory (allocated / reserved) |
|---|---|---|---|---|
| 320x160 | 4 | float32 | 24.3 | 4.7 / 4.8 GB |
| 320x160 | 4 | AMP | 58.6 | 0.9 / 1.2 GB |
| 320x160 | 16 | AMP | 60.6 | 4.2 / 5.1 GB |
| 640x320 | 4 | float32 | 6.0 | 11.9 / 12.9 GB |
| 640x320 | 8 | float32 | 6.3 | 13.0 / 13.7 GB |
| 640x320 | 16 | float32 | out of memory | |
| 640x320 | 4 | AMP | 15.1 | 3.6 / 4.5 GB |
| 640x320 | 16 | AMP | 14.8 | 12.0 / 13.3 GB |

Single-frame inference on the T4 (model only): 13.4 ms (float32) / 5.7 ms (AMP) at 320x160, 57.1 / 20.9 ms at 640x320.
On the desktop CPU the 320x160 model takes about 92 ms per frame (measured earlier); **640x320 on CPU is estimated at
about 4x that (roughly 370 ms), not measured yet.**

## Experiments

### A. Baseline reproduction with the new training loop (seeds 42 and 7)

- **Hypothesis:** the baseline settings with early stopping, best-epoch restore and selection on validation TuSimple
  accuracy reproduce the baseline (validation accuracy about 0.817 single-frame) and stop near epoch 16 instead of 30.
- **Configuration:** `A_baseline_320.yaml`: 320x160, BCE 0.7 + Dice 0.3, AdamW lr 1e-3, weight decay 1e-4, plateau
  schedule, batch 4, no mask dilation, mixed precision (AMP).
- **Benefits / risks:** establishes the reference and the seed-to-seed spread for every later comparison. Risk: a
  different best epoch than the baseline because selection now uses TuSimple accuracy rather than pixel Dice.
- **Evaluation:** validation accuracy within the seed spread of 0.817; the two seeds' difference is the noise floor.

### B. 640x320 input

- **Hypothesis:** at twice the resolution, lanes are 2-4 px wide instead of 1-2 px, so faint and far lanes survive
  downsampling and fewer lanes are missed (lower FN).
- **Configuration:** `B_resolution_640.yaml`: A with `input_size: [640, 320]`; everything else unchanged.
- **Benefits / risks:** more detail for thin lanes. Risks: about 4x the compute per image (training and inference);
  CPU inference latency rises accordingly (the app serves at the checkpoint's size automatically), which may exceed the
  200 ms TuSimple run-time rule on CPU; the receptive field covers less of the scene relative to the image.
- **Evaluation:** validation FN and accuracy against A; CPU latency measured locally before any adoption.

### C. Thicker training targets (C1: 4 px, C2: 8 px each side at 1280x720)

- **Hypothesis:** wider targets (about 1 and 2 px more each side at 320x160) remove the aliasing gaps of 1-2 px lanes, so
  the probability ridge is continuous and lane extraction misses fewer lanes.
- **Configuration:** `C1_mask_dilation_4.yaml`, `C2_mask_dilation_8.yaml`: A with `train_mask_dilation` 4 / 8.
  Validation and test masks are unchanged, so pixel metrics keep one definition.
- **Benefits / risks:** better continuity and recall. Risks: wider predictions lower pixel precision and Dice against
  the thin evaluation masks (expected, and why selection uses lane-level accuracy); lanes close together may merge.
- **Evaluation:** validation accuracy and FN against A; pixel precision reported to show the width trade-off.

### D. Loss functions (D1: focal BCE + Dice, D2: Tversky)

- **Hypothesis:** lane pixels are about 1.3% of each mask; focal BCE (gamma 2, lane weight 0.75) focuses on hard pixels,
  and Tversky (false positives 0.3, false negatives 0.7) penalises missed lane pixels more, both raising recall on faint
  lanes.
- **Configuration:** `D1_focal_dice.yaml`, `D2_tversky.yaml`: A with only the loss changed.
- **Benefits / risks:** fewer missed lanes. Risks: more false positives (FP), noisier probability maps, different
  calibration (the 0.3 extraction threshold may no longer be the best; it is re-checked on validation for the final
  model only).
- **Evaluation:** validation FN and FP against A.

### E. Combination of what helped

- **Rule:** combine only the factors whose validation accuracy beat A by more than A's seed spread (and did not raise FP
  by more than 0.02). If none did, the baseline stays and E is not run.
- **Configuration:** decided from A-D results; trained with seeds 42 and 7.
- **Evaluation:** validation accuracy; then the post-processing grid on validation; then one test run (single-frame and
  with tracking) for the chosen model, compared with the baseline. Only then would it be considered for
  `AI_MODEL_PATH`, after CPU latency is measured.

### Not in the requested list, worth considering: data augmentation

The baseline trains without any augmentation and overfits after epoch 10. Horizontal flips (with lane order unchanged,
since masks are binary) and brightness / contrast changes are standard for TuSimple and need a small implementation in
`transforms.py` first. Proposed as a follow-up experiment F only if approved.

## Targets

Aspirational, not promised: 0.90 official test accuracy (stretch 0.93), lower FN, CPU latency still practical. For
context, published TuSimple results are around 0.96; this project makes no state-of-the-art claim.

## Budget and order

**Estimates**, from the measurements above, all runs with AMP and preloading:

- Per epoch at 320x160, batch 4: 3,268 / 58.6 = 56 s of training compute plus about 20 s of validation (forward pass and
  TuSimple scoring of 351 images on the CPU) = about 75 s (the stage-2 run took 320 s in float32 without preloading).
- Per epoch at 640x320, batch 4: 3,268 / 15.1 = 216 s plus about 40 s of validation = about 255 s.
- Epochs per run: early stopping with patience 6 after a best epoch near 10 (the baseline's) gives about 16; the cap is 30.
- One-time costs per Kaggle session: dataset preparation about 30 min (measured in stage 2), preloading 1-2 min.

| Run | Epochs (likely - cap) | GPU time (likely - cap) |
|---|---|---|
| A, seed 42 | 16 - 30 | 20 - 38 min |
| A, seed 7 | 16 - 30 | 20 - 38 min |
| B (640x320) | 16 - 30 | 68 - 128 min |
| C1, C2, D1, D2 (each) | 16 - 30 | 20 - 38 min |
| Dataset preparation (once per session) | | 30 min |
| **A-D total** | | **about 3.6 - 6.9 GPU hours** |
| E (two seeds, depends on what is combined) | | 0.7 - 4.3 GPU hours |

All of A-D fits in one Kaggle session (12-hour limit) and well within the weekly GPU quota (30 hours at the time of the
baseline run); Kaggle's free GPU quota has no monetary cost. Order within the session: A (seed 42) first, so the
epoch-8 termination rule has its reference, then A (seed 7), C1, C2, D1, D2, and B last (the longest). The early-
termination time limit per epoch is twice the estimate: 150 s for A, C and D, 510 s for B.

The feature branch contains the experiment configs and the [Kaggle preflight workflow](../gpu-training-workflow.md).
Before launching A–D, make the Kaggle runner available in the workspace and verify it pins the code revision, checks
the official split counts (3,268 / 351 / 2,770), and supplies separate `save_dir` and `output_dir` paths for each run,
including the two A seeds. The runner must enforce the per-epoch time limit and epoch-8 comparison and handle the OOM
retry. Keep `evaluate_test: false` and `production_checkpoint_path: null` for every run. Install `matplotlib` in the
Kaggle runtime to generate `curves.png`.

## Decision requested

Approval to run A (two seeds), B, C1, C2, D1 and D2 on Kaggle T4 GPUs as described, within the GPU time above, then E
if justified by the validation results. Each run is a separate, reproducible job from a fixed commit; results are
reported back before E and before any test-split evaluation.
