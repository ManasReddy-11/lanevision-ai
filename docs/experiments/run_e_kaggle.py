"""Paste into a new code cell in the existing LaneVision Kaggle session."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import disk_usage

import matplotlib
import torch


working = Path("/kaggle/working")
workspace = working / "lanevision-gua0a0he"
repo = workspace / "repo"
prepared = workspace / "tusimple_official"
labels = (
    Path("/kaggle/input/datasets/manideep1108/tusimple/TUSimple")
    / "train_set/label_data_0531.json"
)
revision = "4fe9acb1f93e145ae11074ff7c01d7a00982dbd4"
reference_runs = {42: "A-seed42-rqkvsr9j", 7: "A-seed7-9yyou8bj"}

entrypoint = """
import math
import torch
import backend.training.train as training

build_loss = training.build_loss

def guarded_build_loss(config):
    loss_fn = build_loss(config)

    def guarded_loss(logits, targets):
        loss = loss_fn(logits, targets)
        if not math.isfinite(loss.detach().item()):
            raise RuntimeError("Non-finite loss. Stopping experiment.")
        return loss

    return guarded_loss

training.build_loss = guarded_build_loss
try:
    training.main()
except torch.cuda.OutOfMemoryError:
    print("CUDA out of memory", flush=True)
    raise SystemExit(42)
"""


def run_attempt(seed, batch_size, run_dir, reference):
    output = run_dir / f"batch-{batch_size}"
    command = [
        sys.executable, "-u", "-c", entrypoint,
        "--config", "backend/training/configs/experiments/D2_tversky.yaml",
        "--dataset", str(prepared),
        "--batch-size", str(batch_size),
        "--set", "experiment_name=E_dilation_8_tversky",
        "--set", "train_mask_dilation=8",
        "--set", f"seed={seed}",
        "--set", f"tusimple_val_labels={labels}",
        "--set", f"output_dir={output}",
        "--set", f"save_dir={output / 'checkpoints'}",
        "--set", "evaluate_test=false",
        "--set", "production_checkpoint_path=null",
    ]
    with subprocess.Popen(
        command, cwd=repo, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if not line.startswith("epoch="):
                continue
            row = json.loads((output / "metrics.jsonl").read_text().splitlines()[-1])
            epoch = row["epoch"]
            slow = epoch > 2 and row["epoch_seconds"] > 150
            poor = epoch == 8 and row["tusimple_accuracy"] < reference - 0.05
            if slow or poor:
                process.terminate()
                process.wait()
                reason = "epoch exceeded 150 seconds" if slow else "epoch-8 accuracy fell over 5 points below A"
                print(f"E seed {seed} stopped: {reason}. Outputs: {output}", flush=True)
                return None, output
        return process.wait(), output


def main():
    assert torch.cuda.is_available(), "Enable a Kaggle GPU"
    assert prepared.is_dir() and labels.is_file(), "The verified Kaggle dataset is missing"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert head == revision, head

    references = {}
    for seed, folder in reference_runs.items():
        log = working / folder / "batch-4" / "metrics.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        references[seed] = next(row["tusimple_accuracy"] for row in rows if row["epoch"] == 8)

    for seed, reference in references.items():
        assert disk_usage(working).free >= 2 * 2**30, "Free more disk space first"
        run_dir = Path(tempfile.mkdtemp(prefix=f"E-seed{seed}-", dir=working))
        print("Results folder:", run_dir, flush=True)
        status, output = run_attempt(seed, 4, run_dir, reference)
        if status == 42:
            print(f"Retrying seed {seed} at batch size 2 after CUDA OOM.", flush=True)
            status, output = run_attempt(seed, 2, run_dir, reference)
        if status == 0:
            print(f"Completed E seed {seed}:", output / "summary.json", flush=True)
        elif status is not None:
            raise RuntimeError(f"E seed {seed} exited with status {status}. See output above.")


if __name__ == "__main__":
    main()
