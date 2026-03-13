from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from maskdiff.utils import list_images, load_mask, load_rgb, save_json, stem_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted masked faces against paired targets.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-json", help="Optional JSON report path.")
    return parser.parse_args()


def _to_float_rgb(path: Path) -> np.ndarray:
    return np.asarray(load_rgb(path), dtype=np.float32) / 255.0


def _to_float_mask(path: Path) -> np.ndarray:
    return np.asarray(load_mask(path), dtype=np.float32)[..., None] / 255.0


def _resize_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = load_rgb(path).resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _resize_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = load_mask(path).resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.float32)[..., None] / 255.0


def main() -> None:
    args = parse_args()
    pred_index = {
        stem: path
        for stem, path in stem_index(list_images(args.pred_dir)).items()
        if not stem.endswith("_mask") and not stem.endswith("_contact")
    }
    target_index = stem_index(list_images(args.target_dir))
    mask_index = stem_index(list_images(args.mask_dir))

    shared = sorted(set(pred_index) & set(target_index) & set(mask_index))
    if not shared:
        raise SystemExit("No overlapping stems across prediction, target, and mask directories.")

    mae_all = []
    mae_mask = []
    mae_background = []
    psnr = []

    for stem in shared:
        pred = _to_float_rgb(pred_index[stem])
        size = (pred.shape[1], pred.shape[0])
        target = _resize_rgb(target_index[stem], size)
        mask = _resize_mask(mask_index[stem], size)

        if pred.shape != target.shape:
            raise SystemExit(f"Shape mismatch for {stem}: pred {pred.shape} vs target {target.shape}")

        diff = np.abs(pred - target)
        mse = np.mean((pred - target) ** 2)
        mae_all.append(float(diff.mean()))

        mask_sum = max(float(mask.sum()) * pred.shape[2], 1.0)
        background = 1.0 - mask
        background_sum = max(float(background.sum()) * pred.shape[2], 1.0)

        mae_mask.append(float((diff * mask).sum() / mask_sum))
        mae_background.append(float((diff * background).sum() / background_sum))
        psnr.append(float("inf") if mse == 0.0 else 20.0 * math.log10(1.0 / math.sqrt(mse)))

    metrics = {
        "count": len(shared),
        "mae_all": float(np.mean(mae_all)),
        "mae_mask": float(np.mean(mae_mask)),
        "mae_background": float(np.mean(mae_background)),
        "psnr": float(np.mean(psnr)),
    }

    if args.output_json:
        save_json(args.output_json, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
