"""Two-stage inference: mask predictor -> DDPM generator."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from DDPM.core.diffusion import DiffusionSchedule
from DDPM.inference.infer import _ddim_sample
from DDPM.core.model import ConditionalUNet, MaskPredictorUNet
from DDPM.core.utils import (
    choose_device,
    ensure_dir,
    image_to_tensor,
    list_images,
    load_rgb,
    make_contact_sheet,
    tensor_to_image,
    tensor_to_mask_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage inference: mask predictor + DDPM generation.")
    parser.add_argument("--mask-checkpoint", required=True, help="Stage 1 mask predictor checkpoint.")
    parser.add_argument("--ddpm-checkpoint", required=True, help="Stage 2 DDPM generator checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Directory of clean face images.")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated images.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Binarization threshold for predicted mask.")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples (0 = all).")
    parser.add_argument("--save-masks", action="store_true", help="Also save predicted masks.")
    parser.add_argument("--composite-clean-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _load_mask_predictor(checkpoint_path: str, device: torch.device) -> MaskPredictorUNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = MaskPredictorUNet(
        input_channels=config.get("input_channels", 3),
        base_channels=config.get("base_channels", 64),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _load_ddpm(checkpoint_path: str, device: torch.device) -> tuple[ConditionalUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config") or {}
    model = ConditionalUNet(
        input_channels=config.get("input_channels", 7),
        output_channels=config.get("output_channels", 3),
        base_channels=config.get("base_channels", 64),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    state = checkpoint.get("ema_model") or checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    # Load both models
    mask_model = _load_mask_predictor(args.mask_checkpoint, device)
    ddpm_model, ddpm_ckpt = _load_ddpm(args.ddpm_checkpoint, device)

    train_config = ddpm_ckpt.get("train_config") or ddpm_ckpt.get("config", {})
    timesteps = int(train_config.get("timesteps", 250))
    diffusion = DiffusionSchedule(timesteps=timesteps, device=device)

    if args.save_masks:
        mask_output_dir = ensure_dir(output_dir / "predicted_masks")

    input_paths = list_images(args.input_dir)
    if not input_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    limit = len(input_paths) if args.num_samples == 0 else min(args.num_samples, len(input_paths))
    selected = input_paths[:limit]

    for image_path in selected:
        clean_pil = load_rgb(image_path)
        clean_pil = TF.resize(clean_pil, [args.image_size, args.image_size], interpolation=InterpolationMode.BILINEAR)
        clean_tensor = image_to_tensor(clean_pil).unsqueeze(0).to(device)

        # Stage 1: predict binary mask
        with torch.no_grad():
            mask_logits = mask_model(clean_tensor)
            mask_prob = torch.sigmoid(mask_logits)
            mask_tensor = (mask_prob > args.mask_threshold).float()

        # Stage 2: DDPM inpainting — zero mask region so the model fills it in.
        condition = torch.cat([clean_tensor * (1.0 - mask_tensor), mask_tensor], dim=1)
        with torch.no_grad():
            if args.sampler == "ddim":
                generated = _ddim_sample(ddpm_model, diffusion, condition, ddim_steps=args.sample_steps, guidance_scale=args.guidance_scale)
            else:
                generated = diffusion.sample(
                    ddpm_model,
                    condition,
                    guidance_scale=args.guidance_scale,
                    sample_steps=args.sample_steps,
                )
            if args.composite_clean_background:
                generated = generated * mask_tensor + clean_tensor * (1.0 - mask_tensor)

        generated_image = tensor_to_image(generated.squeeze(0))
        generated_image.save(output_dir / f"{image_path.stem}.png")

        if args.save_masks:
            mask_image = tensor_to_mask_image(mask_tensor.squeeze(0))
            mask_image.save(mask_output_dir / f"{image_path.stem}.png")

        print(f"{image_path.stem}: done")

    print(f"Generated {len(selected)} images in {output_dir}")


if __name__ == "__main__":
    main()
