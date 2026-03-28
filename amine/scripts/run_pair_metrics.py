from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import lpips
    import torch
except ModuleNotFoundError:
    lpips = None
    torch = None

try:
    from facemask_utils import derive_binary_mask_from_paths
except ModuleNotFoundError:
    from scripts.facemask_utils import derive_binary_mask_from_paths


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_torch_home() -> Path:
    return Path(__file__).resolve().parents[1] / ".torch_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated masked-face predictions against paired targets."
    )
    parser.add_argument("--pred-dir", required=True, type=Path, help="Directory containing generated images.")
    parser.add_argument("--target-dir", required=True, type=Path, help="Directory containing paired masked RGB targets.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="Optional directory containing paired binary masks.",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        help="Optional directory containing paired clean images. Used to derive masks when --mask-dir is omitted.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=12.0,
        help="Threshold used when deriving evaluation masks from clean and target images.",
    )
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        help="Compute LPIPS in addition to MAE/PSNR. Requires the lpips package.",
    )
    parser.add_argument(
        "--lpips-net",
        default="alex",
        choices=("alex", "vgg", "squeeze"),
        help="Backbone used by the LPIPS metric.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used for LPIPS. Auto prefers CUDA when available.",
    )
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=default_torch_home(),
        help="Directory used by torch/torchvision to cache metric backbone weights.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional output JSON file.")
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def stem_index(paths: list[Path]) -> dict[str, Path]:
    return {path.stem: path for path in paths}


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def to_float_rgb(path: Path) -> np.ndarray:
    return np.asarray(load_rgb(path), dtype=np.float32) / 255.0


def resize_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = load_rgb(path).resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def resize_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = load_mask(path).resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.float32)[..., None] / 255.0


def derive_mask_array(clean_path: Path, target_path: Path, size: tuple[int, int], threshold: float) -> np.ndarray:
    mask = derive_binary_mask_from_paths(clean_path, target_path, threshold)
    mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.float32)[..., None] / 255.0


def to_lpips_tensor(path: Path, size: tuple[int, int]) -> "torch.Tensor":
    image = load_rgb(path).resize(size, resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    args = parse_args()
    args.torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())

    pred_index = {
        stem: path
        for stem, path in stem_index(list_images(args.pred_dir)).items()
        if not stem.endswith("_mask") and not stem.endswith("_contact")
    }
    target_index = stem_index(list_images(args.target_dir))
    clean_index = stem_index(list_images(args.clean_dir)) if args.clean_dir else {}
    mask_index = stem_index(list_images(args.mask_dir)) if args.mask_dir else {}

    shared = sorted(set(pred_index) & set(target_index))
    if args.mask_dir:
        shared = sorted(set(shared) & set(mask_index))
    elif args.clean_dir:
        shared = sorted(set(shared) & set(clean_index))
    else:
        raise SystemExit("Either --mask-dir or --clean-dir must be provided.")
    if not shared:
        raise SystemExit("No overlapping stems across the requested directories.")

    mae_all: list[float] = []
    mae_mask: list[float] = []
    mae_background: list[float] = []
    psnr: list[float] = []
    lpips_values: list[float] = []

    lpips_model = None
    lpips_device = None
    if args.compute_lpips:
        if lpips is None or torch is None:
            raise SystemExit("LPIPS requested, but the lpips package is not installed in the active interpreter.")
        if args.device == "auto":
            lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            lpips_device = torch.device(args.device)
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(lpips_device)
        lpips_model.eval()

    for stem in shared:
        pred = to_float_rgb(pred_index[stem])
        size = (pred.shape[1], pred.shape[0])
        target = resize_rgb(target_index[stem], size)
        if args.mask_dir:
            mask = resize_mask(mask_index[stem], size)
        else:
            mask = derive_mask_array(clean_index[stem], target_index[stem], size, args.mask_threshold)

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

        if lpips_model is not None:
            pred_tensor = to_lpips_tensor(pred_index[stem], size).to(lpips_device)
            target_tensor = to_lpips_tensor(target_index[stem], size).to(lpips_device)
            with torch.no_grad():
                lpips_value = lpips_model(pred_tensor, target_tensor).item()
            lpips_values.append(float(lpips_value))

    metrics = {
        "count": len(shared),
        "mae_all": float(np.mean(mae_all)),
        "mae_mask": float(np.mean(mae_mask)),
        "mae_background": float(np.mean(mae_background)),
        "psnr": float(np.mean(psnr)),
    }
    if lpips_values:
        metrics["lpips"] = float(np.mean(lpips_values))

    print(json.dumps(metrics, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Wrote {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
