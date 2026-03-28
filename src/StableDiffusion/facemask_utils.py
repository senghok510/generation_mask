from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images_recursive(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def index_unique_by_stem(paths: list[Path], label: str) -> dict[str, Path]:
    if not paths:
        raise ValueError(f"No images were found for {label}.")

    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in paths:
        if path.stem in index:
            duplicates.add(path.stem)
        index[path.stem] = path.resolve()

    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        raise ValueError(f"Duplicate stems found in {label}: {sample}")
    return index


def pair_images_by_stem(clean_dir: Path, masked_dir: Path) -> list[tuple[str, Path, Path]]:
    clean_index = index_unique_by_stem(list_images_recursive(clean_dir), f"clean images under {clean_dir}")
    masked_index = index_unique_by_stem(list_images_recursive(masked_dir), f"masked images under {masked_dir}")

    shared = sorted(set(clean_index) & set(masked_index))
    if not shared:
        raise ValueError(f"No paired stems found between {clean_dir} and {masked_dir}")

    return [(stem, clean_index[stem], masked_index[stem]) for stem in shared]


def derive_binary_mask_from_images(clean: Image.Image, masked: Image.Image, threshold: float) -> Image.Image:
    clean_rgb = clean.convert("RGB")
    masked_rgb = masked.convert("RGB")
    if clean_rgb.size != masked_rgb.size:
        masked_rgb = masked_rgb.resize(clean_rgb.size, resample=Image.Resampling.BILINEAR)

    clean_arr = np.asarray(clean_rgb, dtype=np.float32)
    masked_arr = np.asarray(masked_rgb, dtype=np.float32)

    diff = np.abs(masked_arr - clean_arr).max(axis=2)
    binary = (diff > threshold).astype(np.uint8) * 255
    mask = Image.fromarray(binary, mode="L")
    mask = mask.filter(ImageFilter.MinFilter(3))
    mask = mask.filter(ImageFilter.MaxFilter(5))
    mask = mask.filter(ImageFilter.MaxFilter(21))
    mask = mask.filter(ImageFilter.MinFilter(11))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    return mask.point(lambda value: 255 if value > 16 else 0, mode="L")


def derive_binary_mask_from_paths(clean_path: Path, masked_path: Path, threshold: float) -> Image.Image:
    clean = Image.open(clean_path)
    masked = Image.open(masked_path)
    return derive_binary_mask_from_images(clean, masked, threshold)
