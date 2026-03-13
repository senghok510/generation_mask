from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from maskdiff.diffusion import DiffusionSchedule
from maskdiff.model import ConditionalUNet
from maskdiff.utils import (
    choose_device,
    ensure_dir,
    image_to_tensor,
    load_mask,
    load_rgb,
    make_contact_sheet,
    mask_to_tensor,
    tensor_to_image,
    tensor_to_mask_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DDPM reverse sampling from pure noise to final masked image.")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint path.")
    parser.add_argument("--clean-image", required=True, help="Clean conditioning image.")
    parser.add_argument("--mask-image", required=True, help="Binary mask image.")
    parser.add_argument("--output-dir", required=True, help="Directory for reverse-chain outputs.")
    parser.add_argument("--sample-steps", type=int, default=1000, help="Reverse diffusion steps to run.")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="+",
        default=[999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 0],
        help="Timesteps to save as images. Values refer to training-time DDPM indices.",
    )
    parser.add_argument(
        "--composite-clean-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the clean image outside the mask in each saved frame.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _resize_rgb(image: Image.Image, size: int) -> Image.Image:
    return TF.resize(image, [size, size], interpolation=InterpolationMode.BILINEAR)


def _resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return TF.resize(mask, [size, size], interpolation=InterpolationMode.NEAREST)


def _load_model(checkpoint_path: str, device: torch.device) -> tuple[ConditionalUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint.get("train_config") or checkpoint.get("config", {})
    model_config = checkpoint.get("model_config") or {
        "input_channels": 7,
        "output_channels": 3,
        "base_channels": int(train_config.get("base_channels", 64)),
    }
    model = ConditionalUNet(**model_config).to(device)
    state_dict = checkpoint.get("ema_model") or checkpoint["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def _make_row(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    model, checkpoint = _load_model(args.checkpoint, device)
    train_config = checkpoint.get("train_config") or checkpoint.get("config", {})
    timesteps = int(train_config.get("timesteps", 1000))
    image_size = int(train_config.get("image_size", 128))
    diffusion = DiffusionSchedule(timesteps=timesteps, device=device)

    clean = _resize_rgb(load_rgb(args.clean_image), image_size)
    mask = _resize_mask(load_mask(args.mask_image), image_size)
    clean_tensor = image_to_tensor(clean).unsqueeze(0).to(device)
    mask_tensor = mask_to_tensor(mask).unsqueeze(0).to(device)
    condition = torch.cat([clean_tensor, mask_tensor], dim=1)

    current = torch.randn(1, 3, image_size, image_size, device=device)
    total_steps = max(1, min(args.sample_steps, diffusion.timesteps))
    schedule = torch.linspace(diffusion.timesteps - 1, 0, steps=total_steps, device=device).long()
    schedule = torch.unique_consecutive(schedule)
    snapshot_steps = set(int(step) for step in args.snapshot_steps)

    saved_frames: list[tuple[int, Image.Image]] = []
    for index, step in enumerate(schedule.tolist()):
        timestep = torch.full((1,), step, device=device, dtype=torch.long)
        mean, variance, x0 = diffusion.p_mean_variance(model, current, timestep, condition, args.guidance_scale)
        frame = x0
        if args.composite_clean_background:
            frame = frame * mask_tensor + clean_tensor * (1.0 - mask_tensor)

        if step in snapshot_steps:
            frame_image = tensor_to_image(frame.squeeze(0))
            frame_image.save(output_dir / f"t{step:04d}.png")
            saved_frames.append((step, frame_image))

        if index < len(schedule) - 1 and step > 0:
            current = mean + torch.sqrt(variance.clamp(min=1e-20)) * torch.randn_like(current)
        else:
            current = x0

    final = current
    if args.composite_clean_background:
        final = final * mask_tensor + clean_tensor * (1.0 - mask_tensor)

    final_image = tensor_to_image(final.squeeze(0))
    final_image.save(output_dir / "final.png")
    tensor_to_mask_image(mask_tensor.squeeze(0)).save(output_dir / "mask.png")
    clean.save(output_dir / "clean.png")
    make_contact_sheet(clean, tensor_to_mask_image(mask_tensor.squeeze(0)), final_image).save(output_dir / "contact.png")

    ordered = [image for _, image in sorted(saved_frames, key=lambda item: item[0], reverse=True)]
    if ordered:
        _make_row(ordered).save(output_dir / "reverse_chain.png")

    print(f"Saved reverse-chain visualization to {output_dir}")


if __name__ == "__main__":
    main()
