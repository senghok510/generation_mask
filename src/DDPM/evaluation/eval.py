from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from DDPM.core.utils import list_images, load_mask, load_rgb, save_json, stem_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted masks or masked faces.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--mask-dir", help="Binary mask directory (required for image mode).")
    parser.add_argument("--mode", choices=["mask", "image"], default="image",
                        help="'mask' for IoU/Dice on binary masks, 'image' for MAE/PSNR/FID/LPIPS.")
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


def _strip_mask_suffix(stem: str) -> str:
    if stem.endswith("_Mask"):
        return stem[: -len("_Mask")]
    return stem


def _build_stem_index(directory: str, strip_suffixes: bool = False) -> dict[str, Path]:
    paths = list_images(directory)
    if strip_suffixes:
        return {_strip_mask_suffix(p.stem): p for p in paths}
    return stem_index(paths)


# ---- Mask evaluation (IoU / Dice) ----

def eval_masks(pred_dir: str, target_dir: str) -> dict[str, float]:
    pred_index = _build_stem_index(pred_dir)
    target_index = _build_stem_index(target_dir)

    shared = sorted(set(pred_index) & set(target_index))
    if not shared:
        raise SystemExit("No overlapping stems between pred and target mask dirs.")

    ious, dices = [], []
    for stem in shared:
        pred = (np.asarray(load_mask(pred_index[stem]), dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        gt_img = load_mask(target_index[stem])
        # resize GT to match prediction size
        if gt_img.size != (pred.shape[1], pred.shape[0]):
            gt_img = gt_img.resize((pred.shape[1], pred.shape[0]), resample=Image.Resampling.NEAREST)
        gt = (np.asarray(gt_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        ious.append(float(intersection / max(union, 1e-6)))
        dices.append(float(2.0 * intersection / max(pred.sum() + gt.sum(), 1e-6)))

    return {
        "count": len(shared),
        "iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
    }


# ---- Image evaluation (MAE / PSNR / FID / LPIPS) ----

def _compute_fid(pred_dir: str, target_dir: str, shared: list[str],
                 pred_index: dict[str, Path], target_index: dict[str, Path]) -> float:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("torchmetrics[image] not installed, skipping FID.")
        return float("nan")

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
    for stem in shared:
        pred_img = load_rgb(pred_index[stem])
        tgt_img = load_rgb(target_index[stem])
        if tgt_img.size != pred_img.size:
            tgt_img = tgt_img.resize(pred_img.size, resample=Image.Resampling.BILINEAR)
        pred_t = torch.from_numpy(np.asarray(pred_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        tgt_t = torch.from_numpy(np.asarray(tgt_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        fid_metric.update(pred_t, real=False)
        fid_metric.update(tgt_t, real=True)
    return float(fid_metric.compute().item())


def _compute_lpips(pred_dir: str, target_dir: str, shared: list[str],
                   pred_index: dict[str, Path], target_index: dict[str, Path]) -> float:
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError:
        print("torchmetrics[image] not installed, skipping LPIPS.")
        return float("nan")

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    scores = []
    for stem in shared:
        pred_img = load_rgb(pred_index[stem])
        tgt_img = load_rgb(target_index[stem])
        if tgt_img.size != pred_img.size:
            tgt_img = tgt_img.resize(pred_img.size, resample=Image.Resampling.BILINEAR)
        pred_t = torch.from_numpy(np.asarray(pred_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        tgt_t = torch.from_numpy(np.asarray(tgt_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        score = lpips_metric(pred_t, tgt_t)
        scores.append(float(score.item()))
    return float(np.mean(scores))


def eval_images(pred_dir: str, target_dir: str, mask_dir: str) -> dict[str, float]:
    pred_index = {
        stem: path
        for stem, path in _build_stem_index(pred_dir).items()
        if not stem.endswith("_mask") and not stem.endswith("_contact")
    }
    target_index = _build_stem_index(target_dir, strip_suffixes=True)
    mask_index = _build_stem_index(mask_dir)

    shared = sorted(set(pred_index) & set(target_index) & set(mask_index))
    if not shared:
        raise SystemExit("No overlapping stems across prediction, target, and mask directories.")

    mae_all, mae_mask_list, mae_bg_list, psnr_list = [], [], [], []

    for stem in shared:
        pred = _to_float_rgb(pred_index[stem])
        size = (pred.shape[1], pred.shape[0])
        target = _resize_rgb(target_index[stem], size)
        mask = _resize_mask(mask_index[stem], size)

        diff = np.abs(pred - target)
        mse = np.mean((pred - target) ** 2)
        mae_all.append(float(diff.mean()))

        mask_sum = max(float(mask.sum()) * pred.shape[2], 1.0)
        background = 1.0 - mask
        background_sum = max(float(background.sum()) * pred.shape[2], 1.0)

        mae_mask_list.append(float((diff * mask).sum() / mask_sum))
        mae_bg_list.append(float((diff * background).sum() / background_sum))
        psnr_list.append(float("inf") if mse == 0.0 else 20.0 * math.log10(1.0 / math.sqrt(mse)))

    metrics: dict[str, float] = {
        "count": len(shared),
        "mae_all": float(np.mean(mae_all)),
        "mae_mask": float(np.mean(mae_mask_list)),
        "mae_background": float(np.mean(mae_bg_list)),
        "psnr": float(np.mean(psnr_list)),
    }

    # FID and LPIPS (gracefully degrade if torchmetrics not installed)
    metrics["fid"] = _compute_fid(pred_dir, target_dir, shared, pred_index, target_index)
    metrics["lpips"] = _compute_lpips(pred_dir, target_dir, shared, pred_index, target_index)

    return metrics


def main() -> None:
    args = parse_args()

    if args.mode == "mask":
        metrics = eval_masks(args.pred_dir, args.target_dir)
    else:
        if not args.mask_dir:
            raise SystemExit("--mask-dir is required for image evaluation mode.")
        metrics = eval_images(args.pred_dir, args.target_dir, args.mask_dir)

    if args.output_json:
        save_json(args.output_json, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
