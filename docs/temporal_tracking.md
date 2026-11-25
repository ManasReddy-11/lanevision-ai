# Temporal lane tracking and the experimental departure warning

This page describes how the AI video path tracks lanes across frames (`backend/app/ai/lane_tracking.py`), how its
settings were chosen, and what was measured. All numbers come from the baseline checkpoint (TuSimple U-Net, epoch 10,
320x160 input) on CPU. They are measurements unless explicitly marked as estimates.

## What the tracker does

Each frame, `extract_lanes` finds lanes in the model's probability map. In a video, `LaneTracker` (one per video) then:

1. **Associates** each detected lane with a track: the mean horizontal distance between them over the rows both cover
   (or the track's fitted line, when they barely overlap), closest pairs first, within a gate of 4% of the frame width.
2. **Smooths** each track row by row: `x = 0.5 * observed + 0.5 * previous` (an exponential moving average per row).
3. **Remembers rows** a lane was seen on recently (a dash gap, a short occlusion) for 3 frames, shifted with the lane's
   lateral movement, unless they disagree with the lane's current detection by more than the gate.
4. **Confirms** a new lane after 2 consecutive detections, so a one-frame false detection is never shown.
5. **Holds** a lane that is not detected for up to 10 frames, reporting it as held, with its confidence multiplied by
   0.7 per frame. A held lane that crosses an observed lane in the lower half of the frame is not reported.

The ego lane, the vehicle offset and the confidence are then computed from the tracked lanes exactly as for a single
image (`select_ego_lane`). Because a held lane's confidence decays, a held ego boundary supports a reliable offset for
only a frame or two. The ego lane is re-selected every frame, so after a lane change the offset is measured in the new
lane and no geometry of the old one is carried over; the change itself is reported as an event. Tracks are reset for
every video.

In the processed video, lanes held from earlier frames are drawn dashed, and frames whose ego boundary is held carry
the warning `Ego-lane boundary held from previous frames (not detected in this frame)`.

## How it was evaluated

TuSimple provides each labelled frame as the 20th frame of a 20-frame clip (20 fps), and its readme says the clips
"can help algorithms to infer better lane detection results". `backend/training/tusimple_temporal_eval.py`:

- runs the checkpoint on every frame of each clip and extracts lanes with the production settings, sampled at the
  clip's `h_samples` (as the single-frame benchmark does);
- runs each tracker variant over the 20 frames and scores what it reports for the annotated 20th frame with the
  official TuSimple accuracy / FP / FN (without the 200 ms run-time rule; per-frame latency is reported separately);
- measures, over all frames: whether a reliable vehicle offset is available, the absolute change of the offset between
  consecutive frames that both have one ("offset jump", i.e. jitter), how often availability switches on or off,
  lanes per frame, distinct lane identities per clip, ego-lane changes, departure warnings, and tracker latency.

**Settings were chosen on the validation clips only** (`label_data_0531`, the 351 clips of the prepared validation
split), never on the test split. The selection rule, fixed before the second round: highest validation accuracy, with
FP no more than 0.01 above no tracking, offset jitter at most half of no tracking, and offset availability no more than
0.03 below no tracking; ties within 0.003 go to the shorter memory.

*Correction made during selection:* the first round measured availability over all 20 frames. Requiring two detections
before a lane is shown makes frame 1 of every clip unavailable, which costs 0.05 on 20-frame clips by construction and
would exclude every `min_hits = 2` variant, against the rule's intent. Availability was therefore measured on frames
2-20 for the decision; both numbers are reported below.

Rounds (all on validation):

1. One factor at a time around exponential smoothing 0.5: smoothing 0.3/0.5/0.7/1.0, confirmation after 1/2/3
   detections, hold 0/3/6 frames, row memory 0/3/6 frames, and a constant-velocity Kalman filter per row (process noise
   1 or 4, measurement noise 25 or 100 px²).
2. Combinations of the helpful factors (confirmation 1/2 x hold 6/10 x row memory 3/6).
3. The chosen combination with and without the two consistency rules, added after inspecting failure cases (below).

## Results on the validation clips (351 clips, 7,020 frames)

| | Frame-20 accuracy | FP | FN | Offset available (all / frames 2-20) | Availability flips per clip (frames 2-20) | Offset jump mean / median / p95 (px) |
|---|---|---|---|---|---|---|
| No tracking (previous behaviour) | 0.8169 | 0.1569 | 0.2645 | 0.944 / 0.944 | 0.77 | 8.87 / 2.79 / 22.84 |
| **Chosen tracker** | **0.8862** | **0.1571** | **0.1717** | 0.907 / 0.954 | 0.28 | **3.90 / 1.45 / 10.44** |
| Same, without consistency rules | 0.8809 | 0.1583 | 0.1719 | 0.899 / 0.946 | 0.27 | 3.82 / - / 9.31 |
| Exponential smoothing only (round-1 defaults: hold 3) | 0.8507 | 0.1353 | 0.2151 | 0.917 / 0.965 | 0.27 | 3.87 / - / 9.41 |
| Kalman filter per row (4 settings) | 0.8485-0.8492 | 0.1445-0.1468 | 0.2234-0.2251 | 0.916-0.917 / 0.965 | 0.26 | 4.36-5.28 / - / 11.34-13.34 |

The no-tracking row reproduces the single-frame validation benchmark (0.8169), confirming the evaluation path.

- **Why exponential smoothing and not a Kalman filter:** the Kalman variants scored lower on both accuracy and jitter
  than plain per-row exponential smoothing with the same association and memory, so the simpler method was kept.
- **Where the accuracy comes from:** without smoothing (weight 1.0) the tracker still reached 0.850, so most of the gain
  is from association, row memory and holding lanes through missed frames; smoothing is what cuts the jitter.
- With the chosen tracker, frame-20 accuracy is higher than without tracking on 237 of 351 clips, lower on 62 and equal
  on 52 (mean change +0.113 where better, -0.058 where worse).
- Lanes reported per frame: 2.85 (2.79 without tracking); distinct lane identities per clip: 3.33.

## Failure cases (validation)

- **Too many lanes** (found before the consistency rules): a spurious track along a concrete barrier plus two held lanes
  crossing observed ones gave 7 lanes against 3 in the ground truth, and TuSimple scores an image with more than
  ground truth + 2 lanes as 0 (that clip: 0.95 without tracking, 0.00 with). Not reporting held lanes that cross observed
  ones near the vehicle fixed this clip (0.98) and raised validation accuracy from 0.8809 to 0.8862.
- **Mixed tracks** (found before the consistency rules): while a truck occluded the outer lane, a track kept far rows of
  one lane and near rows of another, producing a lane that crosses the ego boundary. Dropping remembered rows that
  disagree with the current detection addresses this (that clip: 0.75 before the rules, 0.89 after; 0.96 without
  tracking).
- **Appearance delay** (accepted): a lane first detected in the last frame is not reported yet, because a new lane needs
  two detections; this costs accuracy on clips where the lane appears late (e.g. 0.97 -> 0.73) and is the price of
  never showing one-frame false detections.
- **Remaining stale lanes:** a held lane can still be drawn for up to 0.5 s (dashed) when it does not cross an observed
  lane; its confidence decays, so it stops supporting the offset quickly.

Example images (single-frame lanes, tracked lanes and ground truth on the 20th frame) are kept outside the repository,
with the evaluation outputs, because they are TuSimple frames.

## Experimental lane departure warning

`LaneDepartureMonitor` warns when |offset| / ego-lane width stays at or above 0.35 for 0.3 s, clears below 0.30, makes no
judgement without a reliable ego lane, and restarts after a change of ego lane.

- On the validation clips it raised 11 warnings (47 frames) in 9 of 351 clips. In the three warnings inspected, the
  camera reference point was on or right next to a lane marking (two were during lane changes the tracker also
  reported).
- **There is no ground truth for departures in TuSimple, so precision and recall are unknown.** The camera reference
  point is assumed to be the frame centre; with no camera calibration, offsets are image pixels, not metres, and a camera
  mounted off-centre shifts every offset (the baseline measured about -100 px on average on TuSimple clips). The warning
  is a demonstration feature, not a driver-assistance or safety function.

## Latency

Tracking, ego-lane selection and the departure monitor took 1.06 ms per frame on an idle desktop CPU before the
consistency rules were added. With the rules, they took 3.9-4.3 ms of CPU time per frame while the CPU was fully loaded
by another job, against 3.1-3.4 ms without the rules under the same load (about 20% more); an idle-CPU measurement of
the final version is still to be done. Model inference dominates: about 92 ms per 1280x720 frame on the same CPU for the
320x160 model.

## Test split

Not measured yet. The one-time test-split run (all 2,782 official test clips, chosen settings versus no tracking) was
interrupted after 953 clips had been cached; `tusimple_temporal_eval cache --skip-existing` resumes it.

## Limitations

- TuSimple clips are 1 second long (20 frames): the measurements cover start-up and short gaps, not long drives, tunnels
  or night driving, and highway scenes only.
- Frame-based settings (2 detections, 10 held frames, 3 frames of row memory) were tuned at 20 fps; at other frame rates
  they correspond to different durations.
- The tracker's held lanes and appearance delay trade accuracy on some clips for stability on most; both effects are
  quantified above.
