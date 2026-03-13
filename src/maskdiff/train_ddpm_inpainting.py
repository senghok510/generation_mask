"""DDPM inpainting training on paired face-mask data.

This variant trains the model to edit locally:
- diffuse a clean-background composite target instead of the raw masked-face image
- weight denoising loss more heavily inside the editable mask region
- penalize reconstruction drift outside the mask region
- optionally prefer best checkpoints by validation FID instead of noise MSE

Data layout on disk (per split):
    <split>/clean/   -> FFHQ original faces         (condition)
    <split>/mask/    -> CMFD faces wearing masks    (target to synthesize)
    <split>/masked/  -> binary segmentation masks   (condition, white = mask region)

Model input:  concat(noisy_target, clean_face, binary_mask) = 7 channels
Model output: predicted noise (3 channels)
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
    """Load (clean_face, masked_face, binary_mask) triplets matched by numeric stem."""

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

        # Random horizontal flip (consistent across all three)
        if self.train and torch.rand(1).item() < 0.5:
            clean = TF.hflip(clean)
            masked_face = TF.hflip(masked_face)
            mask = TF.hflip(mask)

        r = self.image_size
        clean = TF.resize(clean, [r, r], interpolation=InterpolationMode.BILINEAR)
        masked_face = TF.resize(masked_face, [r, r], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [r, r], interpolation=InterpolationMode.NEAREST)

        # Images to [-1, 1], mask to {0, 1}
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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDPM inpainting training.")
    p.add_argument("--train-dir", required=True, help="Train split root.")
    p.add_argument("--val-dir", default=None, help="Validation split root.")
    p.add_argument("--save-dir", default="outputs/ddpm_inpaint_baseline", help="Checkpoint directory.")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--mask-weight", type=float, default=4.0, help="Extra denoising weight inside the editable mask.")
    p.add_argument("--background-weight", type=float, default=2.0, help="Penalty for changing clean pixels outside the mask.")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20, help="Print every N batches.")
    p.add_argument("--save-every", type=int, default=10, help="Write a numbered checkpoint every N epochs.")
    p.add_argument("--fid-every", type=int, default=5, help="Compute FID every N epochs (0 to disable).")
    p.add_argument("--fid-samples", type=int, default=256, help="Max val samples for FID computation.")
    p.add_argument(
        "--best-metric",
        choices=("val_loss", "fid"),
        default="fid",
        help="Metric used to update checkpoint_best.pt when validation is available.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# FID evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_fid(
    ema_model: nn.Module,
    diffusion: DiffusionSchedule,
    val_dl: DataLoader,
    max_samples: int,
    device: torch.device,
) -> float:
    """Generate masked faces with the EMA model and compute FID vs real masked faces."""
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    n_collected = 0
    for batch in val_dl:
        if n_collected >= max_samples:
            break

        clean = batch["clean"].to(device)
        masked_face = batch["masked_face"].to(device)
        mask = batch["mask"].to(device)
        condition = torch.cat([clean, mask], dim=1)

        # Generate via full DDPM reverse sampling (no guidance)
        generated = diffusion.sample(ema_model, condition, guidance_scale=1.0)
        generated = generated * mask + clean * (1.0 - mask)
        target = clean * (1.0 - mask) + masked_face * mask

        # Real and fake to [0, 1] float for normalize=True mode
        real_01 = (target.clamp(-1, 1) + 1.0) * 0.5
        fake_01 = (generated.clamp(-1, 1) + 1.0) * 0.5

        fid.update(real_01, real=True)
        fid.update(fake_01, real=False)
        n_collected += clean.shape[0]

    score = fid.compute().item()
    fid.reset()
    return score


def compute_loss(
    model: nn.Module,
    diffusion: DiffusionSchedule,
    clean: torch.Tensor,
    masked_face: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    mask_weight: float,
    background_weight: float,
) -> torch.Tensor:
    """Train against a clean-background composite target and preserve the background explicitly."""
    target = clean * (1.0 - mask) + masked_face * mask
    noise = torch.randn_like(target)
    t = torch.randint(0, diffusion.timesteps, (target.shape[0],), device=device)
    noisy = diffusion.q_sample(target, t, noise)
    model_input = torch.cat([noisy, clean, mask], dim=1)

    pred_noise = model(model_input, t)
    weights = 1.0 + mask_weight * mask
    noise_loss = ((pred_noise - noise) ** 2 * weights).mean()

    pred_start = diffusion.predict_start_from_noise(noisy, t, pred_noise).clamp(-1.0, 1.0)
    outside = 1.0 - mask
    background_loss = torch.abs((pred_start - clean) * outside).mean()
    return noise_loss + background_weight * background_loss


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
    best_fid = None

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {param_count:,} | Device: {device} | AMP: {amp_enabled}")

    # ---- Loop --------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dl, 1):
            clean = batch["clean"].to(device)              # (B, 3, H, W)
            masked_face = batch["masked_face"].to(device)  # (B, 3, H, W)
            mask = batch["mask"].to(device)                # (B, 1, H, W)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                loss = compute_loss(
                    model=model,
                    diffusion=diffusion,
                    clean=clean,
                    masked_face=masked_face,
                    mask=mask,
                    device=device,
                    mask_weight=args.mask_weight,
                    background_weight=args.background_weight,
                )

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
        status = f"Epoch {epoch}/{args.epochs} | train_loss {avg_train:.6f}"

        # ---- Validation ----------------------------------------------------
        if val_dl is not None:
            ema_model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    clean = batch["clean"].to(device)
                    masked_face = batch["masked_face"].to(device)
                    mask = batch["mask"].to(device)

                    with torch.autocast(device_type="cuda", enabled=amp_enabled):
                        val_total += compute_loss(
                            model=ema_model,
                            diffusion=diffusion,
                            clean=clean,
                            masked_face=masked_face,
                            mask=mask,
                            device=device,
                            mask_weight=args.mask_weight,
                            background_weight=args.background_weight,
                        ).item()

            val_loss = val_total / len(val_dl)
            status += f" | val_loss {val_loss:.6f}"
        else:
            val_loss = None
        fid_score = None

        # ---- FID (expensive: only every N epochs) ----------------------------
        if (
            val_dl is not None
            and args.fid_every > 0
            and (epoch % args.fid_every == 0 or epoch == args.epochs)
        ):
            log.info(f"  Computing FID on {min(args.fid_samples, len(val_ds))} val samples ...")
            fid_score = compute_fid(ema_model, diffusion, val_dl, args.fid_samples, device)
            status += f" | FID {fid_score:.2f}"
            ema_model.eval()

        should_save_best = False
        if val_dl is None:
            should_save_best = True
        elif args.best_metric == "fid" and fid_score is not None:
            if best_fid is None or fid_score < best_fid:
                best_fid = fid_score
                should_save_best = True
        elif val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            should_save_best = True

        if should_save_best:
            torch.save({
                "epoch": epoch,
                "val_loss": val_loss,
                "fid": fid_score,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": {
                    "input_channels": 7,
                    "output_channels": 3,
                    "base_channels": args.base_channels,
                },
                "train_config": vars(args),
            }, output_dir / "checkpoint_best.pt")

        log.info(status)

        # Save last checkpoint every epoch
        torch.save({
            "epoch": epoch, "val_loss": val_loss, "fid": fid_score,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": {
                "input_channels": 7,
                "output_channels": 3,
                "base_channels": args.base_channels,
            },
            "train_config": vars(args),
            "config": vars(args),
        }, output_dir / "checkpoint_last.pt")

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch, "val_loss": val_loss, "fid": fid_score,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": {
                    "input_channels": 7,
                    "output_channels": 3,
                    "base_channels": args.base_channels,
                },
                "train_config": vars(args),
                "config": vars(args),
            }, output_dir / f"checkpoint_epoch_{epoch:03d}.pt")

    log.info(f"Done. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
