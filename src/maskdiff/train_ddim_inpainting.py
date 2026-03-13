"""Baseline DDIM inpainting: train from scratch on paired face-mask data.

Training is identical to DDPM (noise-prediction MSE). The difference is at
inference: DDIM uses a deterministic, accelerated sampler that can generate
in far fewer steps (e.g. 50 instead of 1000).

Data layout on disk (per split):
    <split>/clean/   -> FFHQ original faces        (target to reconstruct)
    <split>/mask/    -> CMFD faces wearing masks    (condition)
    <split>/masked/  -> binary segmentation masks   (condition, white = mask region)

Model input:  concat(noisy_target, masked_face, binary_mask) = 7 channels
Model output: predicted noise (3 channels)
Loss:         MSE(predicted_noise, actual_noise)
Sampling:     DDIM (deterministic, configurable steps via --ddim-steps)
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
from torchmetrics.image.fid import FrechetInceptionDistance

from maskdiff.diffusion import DiffusionSchedule
from maskdiff.model import ConditionalUNet
from maskdiff.utils import choose_device, ensure_dir, save_json, seed_everything

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset (matches current on-disk layout)
# ---------------------------------------------------------------------------

class InpaintDataset(Dataset):
    """Load (clean, masked_face, binary_mask) triplets matched by numeric stem."""

    def __init__(self, root: str | Path, image_size: int = 128, train: bool = True) -> None:
        self.image_size = image_size
        self.train = train
        root = Path(root)

        clean_dir = root / "clean"
        cmfd_dir = root / "mask"
        binary_dir = root / "masked"

        clean_index = {p.stem: p for p in sorted(clean_dir.iterdir()) if p.suffix.lower() == ".png"}
        cmfd_index: dict[str, Path] = {}
        for p in sorted(cmfd_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                cmfd_index[p.stem.replace("_Mask", "")] = p
        binary_index = {p.stem: p for p in sorted(binary_dir.iterdir()) if p.suffix.lower() == ".png"}

        common = sorted(set(clean_index) & set(cmfd_index) & set(binary_index))
        self.samples = [(clean_index[s], cmfd_index[s], binary_index[s]) for s in common]
        if not self.samples:
            raise ValueError(f"No matched triplets under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clean_path, cmfd_path, mask_path = self.samples[idx]

        clean = Image.open(clean_path).convert("RGB")
        masked_face = Image.open(cmfd_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.train and torch.rand(1).item() < 0.5:
            clean = TF.hflip(clean)
            masked_face = TF.hflip(masked_face)
            mask = TF.hflip(mask)

        r = self.image_size
        clean = TF.resize(clean, [r, r], interpolation=InterpolationMode.BILINEAR)
        masked_face = TF.resize(masked_face, [r, r], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [r, r], interpolation=InterpolationMode.NEAREST)

        clean = TF.to_tensor(clean) * 2.0 - 1.0
        masked_face = TF.to_tensor(masked_face) * 2.0 - 1.0
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return {"clean": clean, "masked_face": masked_face, "mask": mask}


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


# ---------------------------------------------------------------------------
# DDIM sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    diffusion: DiffusionSchedule,
    condition: torch.Tensor,
    ddim_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """DDIM sampling (Song et al., 2020). eta=0 is fully deterministic."""
    batch, _, height, width = condition.shape
    device = condition.device

    # Build sub-sequence of timesteps evenly spaced from T-1 down to 0
    total_T = diffusion.timesteps
    step_indices = torch.linspace(total_T - 1, 0, ddim_steps, device=device).long()

    alpha_cumprod = diffusion.alpha_cumprod  # (T,)
    x = torch.randn(batch, 3, height, width, device=device)

    for i in range(len(step_indices)):
        t = step_indices[i]
        t_batch = t.expand(batch)

        # Predict noise
        model_input = torch.cat([x, condition], dim=1)
        pred_noise = model(model_input, t_batch)

        # Predict x0 from noise
        alpha_t = alpha_cumprod[t]
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()
        x0_pred = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        if i < len(step_indices) - 1:
            t_prev = step_indices[i + 1]
            alpha_t_prev = alpha_cumprod[t_prev]
        else:
            # Last step: go to t=0
            alpha_t_prev = torch.tensor(1.0, device=device)

        # DDIM update rule
        # sigma = eta * sqrt((1 - alpha_{t-1}) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_{t-1})
        sigma = eta * (
            ((1.0 - alpha_t_prev) / (1.0 - alpha_t)).sqrt()
            * (1.0 - alpha_t / alpha_t_prev).clamp(min=0).sqrt()
        )

        # Direction pointing to x_t
        dir_xt = (1.0 - alpha_t_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise

        # x_{t-1}
        x = alpha_t_prev.sqrt() * x0_pred + dir_xt
        if sigma > 0 and i < len(step_indices) - 1:
            x = x + sigma * torch.randn_like(x)

    return x0_pred


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline DDIM inpainting training.")
    p.add_argument("--train-dir", required=True, help="Train split root.")
    p.add_argument("--val-dir", default=None, help="Validation split root.")
    p.add_argument("--save-dir", default="outputs/ddim_inpaint_baseline", help="Checkpoint directory.")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=1000, help="Training diffusion timesteps.")
    p.add_argument("--ddim-steps", type=int, default=50, help="DDIM sampling steps for FID evaluation.")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0=deterministic, 1=DDPM).")
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20, help="Print every N batches.")
    p.add_argument("--fid-every", type=int, default=5, help="Compute FID every N epochs (0 to disable).")
    p.add_argument("--fid-samples", type=int, default=256, help="Max val samples for FID computation.")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# FID evaluation (uses DDIM sampler)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_fid(
    ema_model: nn.Module,
    diffusion: DiffusionSchedule,
    val_dl: DataLoader,
    max_samples: int,
    device: torch.device,
    ddim_steps: int = 50,
    eta: float = 0.0,
) -> float:
    """Generate inpainted images with DDIM sampling and compute FID vs real clean faces."""
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    n_collected = 0
    for batch in val_dl:
        if n_collected >= max_samples:
            break

        clean = batch["clean"].to(device)
        masked_face = batch["masked_face"].to(device)
        mask = batch["mask"].to(device)
        condition = torch.cat([masked_face, mask], dim=1)

        generated = ddim_sample(ema_model, diffusion, condition, ddim_steps=ddim_steps, eta=eta)

        real_01 = (clean.clamp(-1, 1) + 1.0) * 0.5
        fake_01 = (generated.clamp(-1, 1) + 1.0) * 0.5

        fid.update(real_01, real=True)
        fid.update(fake_01, real=False)
        n_collected += clean.shape[0]

    score = fid.compute().item()
    fid.reset()
    return score


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = ensure_dir(args.save_dir)
    seed_everything(args.seed)

    # ---- Logging -----------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )

    config = vars(args)
    save_json(output_dir / "config.json", config)
    log.info("=" * 60)
    log.info("Training config:")
    for k, v in config.items():
        log.info(f"  {k}: {v}")
    log.info("=" * 60)

    # ---- Data --------------------------------------------------------------
    train_ds = InpaintDataset(args.train_dir, image_size=args.image_size, train=True)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = None
    if args.val_dir:
        val_ds = InpaintDataset(args.val_dir, image_size=args.image_size, train=False)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    log.info(f"Train: {len(train_ds)} samples, {len(train_dl)} batches/epoch")
    if val_dl:
        log.info(f"Val:   {len(val_ds)} samples")

    # ---- Model + diffusion -------------------------------------------------
    model = ConditionalUNet(
        input_channels=7, output_channels=3, base_channels=args.base_channels,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    diffusion = DiffusionSchedule(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val = None
    start_epoch = 1

    # ---- Resume from checkpoint --------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            # Apply new lr to all param groups
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
                pg["weight_decay"] = args.weight_decay
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_loss")
        log.info(f"Resumed from {args.resume} (epoch {start_epoch - 1})")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {param_count:,} | Device: {device} | AMP: {amp_enabled}")
    log.info(f"DDIM inference: {args.ddim_steps} steps, eta={args.eta}")

    # ---- Loop --------------------------------------------------------------
    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dl, 1):
            clean = batch["clean"].to(device)
            masked_face = batch["masked_face"].to(device)
            mask = batch["mask"].to(device)

            # Forward diffusion on the clean target (same as DDPM)
            noise = torch.randn_like(clean)
            t = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device)
            noisy = diffusion.q_sample(clean, t, noise)

            model_input = torch.cat([noisy, masked_face, mask], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                pred_noise = model(model_input, t)
                loss = nn.functional.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            update_ema(ema_model, model, args.ema_decay)
            epoch_loss += loss.item()

            if step % args.log_every == 0 or step == len(train_dl):
                log.info(f"  epoch {epoch} | step {step}/{len(train_dl)} | loss {epoch_loss / step:.6f}")

        avg_train = epoch_loss / len(train_dl)
        status = f"Epoch {epoch}/{end_epoch} | train_loss {avg_train:.6f}"

        # ---- Validation (noise-prediction MSE on EMA model) ----------------
        if val_dl is not None:
            ema_model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    clean = batch["clean"].to(device)
                    masked_face = batch["masked_face"].to(device)
                    mask = batch["mask"].to(device)

                    noise = torch.randn_like(clean)
                    t = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device)
                    noisy = diffusion.q_sample(clean, t, noise)
                    model_input = torch.cat([noisy, masked_face, mask], dim=1)

                    with torch.autocast(device_type="cuda", enabled=amp_enabled):
                        pred_noise = ema_model(model_input, t)
                        val_total += nn.functional.mse_loss(pred_noise, noise).item()

            val_loss = val_total / len(val_dl)
            status += f" | val_loss {val_loss:.6f}"

            if best_val is None or val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "epoch": epoch, "val_loss": val_loss,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "config": vars(args),
                }, output_dir / "checkpoint_best.pt")
        else:
            val_loss = None

        # ---- FID (uses fast DDIM sampling) ---------------------------------
        if (
            args.fid_every > 0
            and (epoch % args.fid_every == 0 or epoch == end_epoch)
        ):
            ema_model.eval()
            # Train FID
            log.info(f"  Computing Train FID ({args.ddim_steps}-step DDIM) on {min(args.fid_samples, len(train_ds))} samples ...")
            train_fid = compute_fid(
                ema_model, diffusion, train_dl, args.fid_samples, device,
                ddim_steps=args.ddim_steps, eta=args.eta,
            )
            status += f" | train_FID {train_fid:.2f}"
            # Test FID
            if val_dl is not None:
                log.info(f"  Computing Test FID ({args.ddim_steps}-step DDIM) on {min(args.fid_samples, len(val_ds))} samples ...")
                test_fid = compute_fid(
                    ema_model, diffusion, val_dl, args.fid_samples, device,
                    ddim_steps=args.ddim_steps, eta=args.eta,
                )
                status += f" | test_FID {test_fid:.2f}"

        log.info(status)

        # Save last checkpoint every epoch
        torch.save({
            "epoch": epoch, "val_loss": val_loss,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
        }, output_dir / "checkpoint_last.pt")

    log.info(f"Done. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
