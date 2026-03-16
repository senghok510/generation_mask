from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from DDPM.core.model import MaskPredictorUNet
from DDPM.core.utils import (
    choose_device,
    ensure_dir,
    image_to_tensor,
    list_images,
    load_rgb,
    tensor_to_mask_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mask predictor inference on clean face images.")
    parser.add_argument("--checkpoint", required=True, help="Mask predictor checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Directory of clean face images.")
    parser.add_argument("--output-dir", required=True, help="Directory to write predicted binary masks.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Binarization threshold.")
    parser.add_argument("--num-samples", type=int, default=0, help="How many samples to run. Use 0 for all.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _load_model(checkpoint_path: str, device: torch.device) -> tuple[MaskPredictorUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = MaskPredictorUNet(
        input_channels=int(config.get("input_channels", 3)),
        base_channels=int(config.get("base_channels", 64)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    model, checkpoint = _load_model(args.checkpoint, device)
    train_config = checkpoint.get("train_config") or {}
    image_size = int(train_config.get("image_size", args.image_size))

    input_paths = list_images(args.input_dir)
    if not input_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    limit = len(input_paths) if args.num_samples == 0 else min(args.num_samples, len(input_paths))
    selected = input_paths[:limit]

    for image_path in selected:
        clean = load_rgb(image_path)
        clean = TF.resize(clean, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
        clean_tensor = image_to_tensor(clean).unsqueeze(0).to(device)

        logits = model(clean_tensor)
        mask_tensor = (torch.sigmoid(logits) > args.mask_threshold).float()
        tensor_to_mask_image(mask_tensor.squeeze(0)).save(output_dir / f"{image_path.stem}.png")
        print(f"{image_path.stem}: saved mask")

    print(f"Wrote {len(selected)} predicted masks to {output_dir}")


if __name__ == "__main__":
    main()
