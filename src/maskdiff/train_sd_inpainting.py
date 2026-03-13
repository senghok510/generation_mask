"""Fine-tune stable-diffusion-v1-5/stable-diffusion-inpainting on paired face-mask data.

Data layout expected on disk (per split):
    <split>/clean/   -> FFHQ original faces  (target)
    <split>/mask/    -> CMFD faces with mask  (condition / input image)
    <split>/masked/  -> binary segmentation   (inpainting mask, white = inpaint region)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionInpaintPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InpaintingDataset(Dataset):
    """Yields (target, masked_image, mask) triplets matched by filename number."""

    def __init__(self, root: str | Path, resolution: int = 512) -> None:
        self.resolution = resolution
        root = Path(root)

        clean_dir = root / "clean"
        cmfd_dir = root / "mask"
        binary_dir = root / "masked"

        # Index files by their numeric stem
        clean_index = {p.stem: p for p in sorted(clean_dir.iterdir()) if p.suffix.lower() == ".png"}
        cmfd_index: dict[str, Path] = {}
        for p in sorted(cmfd_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                cmfd_index[p.stem.replace("_Mask", "")] = p
        binary_index = {p.stem: p for p in sorted(binary_dir.iterdir()) if p.suffix.lower() == ".png"}

        # Keep only triplets where all three exist
        common = sorted(set(clean_index) & set(cmfd_index) & set(binary_index))
        self.samples = [
            (clean_index[s], cmfd_index[s], binary_index[s]) for s in common
        ]
        if not self.samples:
            raise ValueError(f"No matched triplets found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clean_path, cmfd_path, mask_path = self.samples[idx]

        target = Image.open(clean_path).convert("RGB")
        masked_image = Image.open(cmfd_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Random horizontal flip (same for all three)
        if torch.rand(1).item() < 0.5:
            target = TF.hflip(target)
            masked_image = TF.hflip(masked_image)
            mask = TF.hflip(mask)

        r = self.resolution
        target = TF.resize(target, [r, r], interpolation=InterpolationMode.BILINEAR)
        masked_image = TF.resize(masked_image, [r, r], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [r, r], interpolation=InterpolationMode.NEAREST)

        # Normalize images to [-1, 1], mask to {0, 1}
        target = TF.to_tensor(target) * 2.0 - 1.0
        masked_image = TF.to_tensor(masked_image) * 2.0 - 1.0
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return {"target": target, "masked_image": masked_image, "mask": mask}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SD inpainting on face-mask data.")
    p.add_argument("--train-dir", type=str, required=True, help="Train split root (contains clean/, mask/, masked/).")
    p.add_argument("--val-dir", type=str, default=None, help="Validation split root.")
    p.add_argument("--output-dir", type=str, default="checkpoints/sd-inpaint", help="Where to save checkpoints.")
    p.add_argument("--model-id", type=str, default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    p.add_argument("--log-every", type=int, default=50, help="Log loss every N steps.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load pretrained components ----------------------------------------
    print(f"Loading model: {args.model_id}")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # Freeze VAE and text encoder — only fine-tune the UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)
    unet.to(device, dtype=torch.float32)
    unet.train()

    # Precompute empty-prompt text embedding (used for all samples)
    text_input = tokenizer(
        "", padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",
    )
    with torch.no_grad():
        empty_text_emb = text_encoder(text_input.input_ids.to(device))[0]  # (1, 77, 768)

    # ---- Datasets ----------------------------------------------------------
    train_ds = InpaintingDataset(args.train_dir, resolution=args.resolution)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"Train samples: {len(train_ds)} | Steps/epoch: {len(train_dl)}")

    val_dl = None
    if args.val_dir:
        val_ds = InpaintingDataset(args.val_dir, resolution=args.resolution)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print(f"Val samples: {len(val_ds)}")

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ---- Training loop -----------------------------------------------------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dl):
            target = batch["target"].to(device)            # (B, 3, H, W) [-1,1]
            masked_image = batch["masked_image"].to(device) # (B, 3, H, W) [-1,1]
            mask = batch["mask"].to(device)                 # (B, 1, H, W) {0,1}

            with torch.no_grad():
                # Encode target → latent
                latents = vae.encode(target.half()).latent_dist.sample().float()
                latents = latents * vae.config.scaling_factor

                # Encode masked image → latent (condition)
                masked_latents = vae.encode(masked_image.half()).latent_dist.sample().float()
                masked_latents = masked_latents * vae.config.scaling_factor

                # Downsample mask to latent spatial size
                mask_latent = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")

            # Forward diffusion: add noise to target latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device, dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # SD inpainting UNet input: 9 channels = noisy(4) + mask(1) + condition(4)
            unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)
            encoder_hidden_states = empty_text_emb.expand(latents.shape[0], -1, -1)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                noise_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / (step + 1)
                print(f"  epoch {epoch} | step {global_step} | loss {avg:.6f}")

        avg_train = epoch_loss / len(train_dl)
        status = f"Epoch {epoch}/{args.epochs} | train_loss {avg_train:.6f}"

        # ---- Validation ----------------------------------------------------
        if val_dl is not None:
            unet.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    target = batch["target"].to(device)
                    masked_image = batch["masked_image"].to(device)
                    mask = batch["mask"].to(device)

                    latents = vae.encode(target.half()).latent_dist.sample().float()
                    latents = latents * vae.config.scaling_factor
                    masked_latents = vae.encode(masked_image.half()).latent_dist.sample().float()
                    masked_latents = masked_latents * vae.config.scaling_factor
                    mask_latent = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=device, dtype=torch.long,
                    )
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)
                    encoder_hidden_states = empty_text_emb.expand(latents.shape[0], -1, -1)

                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        noise_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
                        val_loss += F.mse_loss(noise_pred, noise).item()

            val_loss /= len(val_dl)
            status += f" | val_loss {val_loss:.6f}"
            unet.train()

        print(status)

        # ---- Checkpoint ----------------------------------------------------
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_dir = output_dir / f"epoch-{epoch}"
            unet.save_pretrained(ckpt_dir / "unet")
            print(f"  -> saved {ckpt_dir}")

    # Save final pipeline for easy inference
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipeline.save_pretrained(output_dir / "pipeline-final")
    print(f"Final pipeline saved to {output_dir / 'pipeline-final'}")


if __name__ == "__main__":
    main()
