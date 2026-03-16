from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from DDPM.core.diffusion import DiffusionSchedule
from DDPM.core.model import ConditionalUNet
from DDPM.core.utils import (
    choose_device,
    ensure_dir,
    image_to_tensor,
    list_images,
    load_mask,
    load_rgb,
    make_contact_sheet,
    mask_to_tensor,
    tensor_to_image,
    tensor_to_mask_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on clean faces with binary masks using a trained diffusion checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", help="Single clean image path or directory of clean face images.")
    parser.add_argument("--mask-dir", help="Directory of binary masks matched by stem.")
    parser.add_argument("--test-dir", help="Dataset root containing clean/, mask/ and masked/ subdirectories.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--sample-steps", type=int, default=100, help="Reverse diffusion steps to use at inference time.")
    parser.add_argument("--num-samples", type=int, default=5, help="How many samples to run. Use 0 for all.")
    parser.add_argument("--visualize-count", type=int, default=5, help="How many samples to place in the summary row.")
    parser.add_argument(
        "--composite-clean-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the clean image outside the binary mask after sampling.",
    )
    parser.add_argument("--figure-caption", default="", help="Optional caption to render under the summary row.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _load_inputs(path: str) -> list[Path]:
    candidate = Path(path)
    if candidate.is_dir():
        images = list_images(candidate)
        if not images:
            raise SystemExit(f"No input images found in {candidate}")
        return images
    if not candidate.exists():
        raise SystemExit(f"Input path does not exist: {candidate}")
    return [candidate]


def _resize_rgb(image: Image.Image, size: int) -> Image.Image:
    return TF.resize(image, [size, size], interpolation=InterpolationMode.BILINEAR)


def _resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return TF.resize(mask, [size, size], interpolation=InterpolationMode.NEAREST)


def _mask_stem(stem: str) -> str:
    return stem if stem.endswith("_Mask") else f"{stem}_Mask"


def _resolve_condition_paths(args: argparse.Namespace) -> tuple[list[Path], Path]:
    if args.test_dir:
        test_dir = Path(args.test_dir)
        input_dir = test_dir / "clean"
        mask_dir = test_dir / "masked"
    else:
        if not args.input or not args.mask_dir:
            raise SystemExit("Provide either --test-dir or both --input and --mask-dir.")
        input_dir = Path(args.input)
        mask_dir = Path(args.mask_dir)

    input_paths = _load_inputs(str(input_dir))
    return input_paths, mask_dir


def _load_checkpoint_model(checkpoint_path: str, device: torch.device) -> tuple[ConditionalUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = checkpoint.get("model_config") or {
        "input_channels": int(config.get("input_channels", 7)),
        "output_channels": int(config.get("output_channels", 3)),
        "base_channels": int(config.get("base_channels", 64)),
    }
    model = ConditionalUNet(**model_config).to(device)
    state = checkpoint.get("ema_model") or checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def _make_summary_row(contacts: list[Image.Image]) -> Image.Image:
    if not contacts:
        raise ValueError("No contact sheets available for visualization.")
    width = sum(image.width for image in contacts)
    height = max(image.height for image in contacts)
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    x = 0
    for image in contacts:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _make_image_row(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("No images available for visualization.")
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _load_caption_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        width = draw.textbbox((0, 0), trial, font=font)[2]
        if width <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _make_captioned_figure(images: list[Image.Image], caption: str) -> Image.Image:
    row = _make_image_row(images)
    if not caption:
        return row

    margin_x = 36
    margin_top = 24
    gap = 28
    margin_bottom = 28
    font = _load_caption_font(size=24)
    figure_width = row.width + margin_x * 2
    scratch = Image.new("RGB", (figure_width, 10), color="white")
    scratch_draw = ImageDraw.Draw(scratch)
    lines = _wrap_text(scratch_draw, caption, font, row.width)
    line_heights = [scratch_draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    caption_height = sum(line_heights) + max(0, len(lines) - 1) * 10

    figure_height = margin_top + row.height + gap + caption_height + margin_bottom
    figure = Image.new("RGB", (figure_width, figure_height), color="white")
    figure.paste(row, (margin_x, margin_top))

    draw = ImageDraw.Draw(figure)
    y = margin_top + row.height + gap
    for line, height in zip(lines, line_heights):
        draw.text((margin_x, y), line, fill="black", font=font)
        y += height + 10
    return figure


@torch.no_grad()
def _ddim_sample(
    model: ConditionalUNet,
    diffusion: DiffusionSchedule,
    condition: torch.Tensor,
    ddim_steps: int,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    batch, _, height, width = condition.shape
    device = condition.device
    step_indices = torch.linspace(diffusion.timesteps - 1, 0, ddim_steps, device=device).long()
    x = torch.randn(batch, 3, height, width, device=device)
    do_cfg = guidance_scale > 1.0

    for i in range(len(step_indices)):
        t = step_indices[i]
        t_batch = t.expand(batch)

        if do_cfg:
            uncond = torch.zeros_like(condition)
            pred_uncond = model(torch.cat([x, uncond], dim=1), t_batch)
            pred_cond   = model(torch.cat([x, condition], dim=1), t_batch)
            pred_noise  = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_noise = model(torch.cat([x, condition], dim=1), t_batch)

        alpha_t = diffusion.alpha_cumprod[t]
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()
        x0_pred = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        if i < len(step_indices) - 1:
            alpha_prev = diffusion.alpha_cumprod[step_indices[i + 1]]
            x = alpha_prev.sqrt() * x0_pred + (1.0 - alpha_prev).sqrt() * pred_noise
        else:
            x = x0_pred

    return x


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    input_paths, mask_dir = _resolve_condition_paths(args)

    model, checkpoint = _load_checkpoint_model(args.checkpoint, device)
    train_config = checkpoint.get("train_config") or checkpoint.get("config", {})
    timesteps = int(train_config.get("timesteps", 250))
    image_size = int(train_config.get("image_size", args.image_size))
    diffusion = DiffusionSchedule(timesteps=timesteps, device=device)
    limit = len(input_paths) if args.num_samples == 0 else min(args.num_samples, len(input_paths))
    selected_paths = input_paths[:limit]
    summary_generated: list[Image.Image] = []

    for image_path in selected_paths:
        clean = load_rgb(image_path)
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise SystemExit(f"Missing binary mask for {image_path.stem}: {mask_path}")
        mask = load_mask(mask_path)

        clean = _resize_rgb(clean, image_size)
        mask = _resize_mask(mask, image_size)

        clean_tensor = image_to_tensor(clean).unsqueeze(0).to(device)
        mask_tensor = mask_to_tensor(mask).unsqueeze(0).to(device)
        # Inpainting condition: zero the mask region so the model fills it in.
        condition = torch.cat([clean_tensor * (1.0 - mask_tensor), mask_tensor], dim=1)

        with torch.no_grad():
            if args.sampler == "ddim":
                generated = _ddim_sample(model, diffusion, condition, ddim_steps=args.sample_steps, guidance_scale=args.guidance_scale)
            else:
                generated = diffusion.sample(
                    model,
                    condition,
                    guidance_scale=args.guidance_scale,
                    sample_steps=args.sample_steps,
                )
            if args.composite_clean_background:
                generated = generated * mask_tensor + clean_tensor * (1.0 - mask_tensor)

        generated_image = tensor_to_image(generated.squeeze(0))
        mask_image = tensor_to_mask_image(mask_tensor.squeeze(0))
        contact = make_contact_sheet(clean, mask_image, generated_image)

        generated_image.save(output_dir / f"{image_path.stem}.png")
        print(f"{image_path.stem}: saved prediction")

        if len(summary_generated) < args.visualize_count:
            summary_generated.append(generated_image)

    if summary_generated:
        summary = _make_image_row(summary_generated)
        summary_path = output_dir / f"summary_row_{len(summary_generated)}.png"
        summary.save(summary_path)
        print(f"Saved summary row to {summary_path}")
        if args.figure_caption:
            figure = _make_captioned_figure(summary_generated, args.figure_caption)
            figure_path = output_dir / f"figure_row_{len(summary_generated)}.png"
            figure.save(figure_path)
            print(f"Saved captioned figure to {figure_path}")

    print(f"Wrote {len(selected_paths)} generated images to {output_dir}")


if __name__ == "__main__":
    main()
