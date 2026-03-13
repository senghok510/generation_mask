"""Run a sequence of training experiments with varying hyperparameters.

Each experiment can either start fresh or resume from a previous checkpoint.
After all experiments, a summary table with Train/Test FID is printed.

Usage:
    PYTHONPATH=src python3 -m maskdiff.run_experiments
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Experiment:
    name: str
    epochs: int
    lr: float
    weight_decay: float
    resume_from: str | None = None  # path to checkpoint, or experiment name
    extra_args: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Define your experiment schedule here
# ---------------------------------------------------------------------------

TRAIN_DIR = "data/train"
VAL_DIR = "data/test"
BASE_DIR = "outputs/experiments"

# Common args shared by all experiments
COMMON = dict(
    image_size="128",
    batch_size="8",
    timesteps="1000",
    ddim_steps="50",
    eta="0.0",
    fid_every="0",          # skip per-epoch FID during training (eval at end)
    fid_samples="256",
    base_channels="64",
)

EXPERIMENTS = [
    Experiment(
        name="01_initial",
        epochs=10,
        lr=2e-4,
        weight_decay=0.5,
    ),
    Experiment(
        name="02_reduced_wd",
        epochs=7,
        lr=2e-4,
        weight_decay=0.005,
        resume_from="01_initial",
    ),
    Experiment(
        name="03_continued",
        epochs=25,
        lr=2e-4,
        weight_decay=0.005,
        resume_from="02_reduced_wd",
    ),
    Experiment(
        name="04_continued_2",
        epochs=10,
        lr=2e-4,
        weight_decay=0.005,
        resume_from="03_continued",
    ),
    Experiment(
        name="05_reduced_lr",
        epochs=20,
        lr=2.5e-5,
        weight_decay=0.005,
        resume_from="04_continued_2",
    ),
    Experiment(
        name="06_reduced_wd_01",
        epochs=30,
        lr=2.5e-5,
        weight_decay=0.1,
        resume_from="05_reduced_lr",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def resolve_checkpoint(exp: Experiment) -> str | None:
    if exp.resume_from is None:
        return None
    # If it's a raw path, use it directly
    p = Path(exp.resume_from)
    if p.exists():
        return str(p)
    # Otherwise treat as experiment name
    ckpt = Path(BASE_DIR) / exp.resume_from / "checkpoint_last.pt"
    if ckpt.exists():
        return str(ckpt)
    raise FileNotFoundError(f"Cannot find checkpoint for '{exp.resume_from}': tried {ckpt}")


def run_training(exp: Experiment) -> None:
    save_dir = f"{BASE_DIR}/{exp.name}"
    cmd = [
        sys.executable, "-m", "maskdiff.train_ddim_inpainting",
        "--train-dir", TRAIN_DIR,
        "--val-dir", VAL_DIR,
        "--save-dir", save_dir,
        "--epochs", str(exp.epochs),
        "--lr", str(exp.lr),
        "--weight-decay", str(exp.weight_decay),
    ]
    for k, v in {**COMMON, **exp.extra_args}.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]

    resume_path = resolve_checkpoint(exp)
    if resume_path:
        cmd += ["--resume", resume_path]

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp.name}")
    print(f"  epochs={exp.epochs}  lr={exp.lr}  wd={exp.weight_decay}")
    if resume_path:
        print(f"  resume={resume_path}")
    print(f"{'='*70}\n")

    subprocess.run(cmd, check=True)


def evaluate_fid(exp_name: str) -> tuple[float, float]:
    """Evaluate train + test FID for a finished experiment."""
    save_dir = Path(BASE_DIR) / exp_name
    ckpt_path = save_dir / "checkpoint_last.pt"

    cmd = [
        sys.executable, "-m", "maskdiff.eval_fid",
        "--checkpoint", str(ckpt_path),
        "--train-dir", TRAIN_DIR,
        "--test-dir", VAL_DIR,
        "--fid-samples", COMMON["fid_samples"],
        "--ddim-steps", COMMON["ddim_steps"],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        scores = json.loads(result.stdout)
        return scores["train_fid"], scores["test_fid"]
    else:
        print(f"  FID eval failed for {exp_name}: {result.stderr}")
        return float("nan"), float("nan")


def print_table(results: list[tuple[str, int, float, float]]) -> None:
    header = f"{'Experiment':<30} {'Epochs':>8} {'Train FID':>12} {'Test FID':>12}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, epochs, train_fid, test_fid in results:
        print(f"{name:<30} {epochs:>8} {train_fid:>12.2f} {test_fid:>12.2f}")
    print(sep)


def main() -> None:
    results = []
    cumulative_epochs = 0

    for exp in EXPERIMENTS:
        run_training(exp)
        cumulative_epochs += exp.epochs

        # Quick FID eval using the training script with fid_every=1, epochs=0 trick
        # Instead, just re-use compute_fid directly
        train_fid, test_fid = eval_fid_direct(exp.name)
        results.append((exp.name, cumulative_epochs, train_fid, test_fid))
        print_table(results)

    print("\n\nFINAL RESULTS:")
    print_table(results)


def eval_fid_direct(exp_name: str) -> tuple[float, float]:
    """Compute FID directly in-process (avoids separate script)."""
    import torch
    from maskdiff.train_ddim_inpainting import (
        InpaintDataset, compute_fid, ddim_sample,
    )
    from maskdiff.diffusion import DiffusionSchedule
    from maskdiff.model import ConditionalUNet
    from maskdiff.utils import choose_device

    device = choose_device("auto")
    save_dir = Path(BASE_DIR) / exp_name
    ckpt = torch.load(save_dir / "checkpoint_last.pt", map_location=device, weights_only=True)

    cfg = ckpt.get("config", {})
    model = ConditionalUNet(
        input_channels=7, output_channels=3,
        base_channels=cfg.get("base_channels", 64),
    ).to(device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()

    timesteps = cfg.get("timesteps", 1000)
    diffusion = DiffusionSchedule(timesteps=timesteps, device=device)
    ddim_steps = int(COMMON["ddim_steps"])
    fid_samples = int(COMMON["fid_samples"])
    image_size = int(COMMON["image_size"])

    from torch.utils.data import DataLoader

    print(f"  Evaluating FID for {exp_name} ...")

    train_ds = InpaintDataset(TRAIN_DIR, image_size=image_size, train=False)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=4)
    train_fid = compute_fid(model, diffusion, train_dl, fid_samples, device, ddim_steps=ddim_steps)

    test_ds = InpaintDataset(VAL_DIR, image_size=image_size, train=False)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)
    test_fid = compute_fid(model, diffusion, test_dl, fid_samples, device, ddim_steps=ddim_steps)

    print(f"  {exp_name}: train_FID={train_fid:.2f}, test_FID={test_fid:.2f}")
    return train_fid, test_fid


if __name__ == "__main__":
    main()
