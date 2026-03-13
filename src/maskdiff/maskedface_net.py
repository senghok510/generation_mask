from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from maskdiff.utils import ensure_dir, list_images_recursive, save_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare paired FFHQ/CMFD data for maskdiff training.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser("prepare", help="Pair FFHQ/CMFD images and build train/test splits.")
    prepare.add_argument("--ffhq-dir", required=True, help="Directory containing clean FFHQ images.")
    prepare.add_argument("--cmfd-dir", required=True, help="Directory containing MaskedFace-Net CMFD images.")
    prepare.add_argument("--output-dir", required=True, help="Output root for train/test splits.")
    prepare.add_argument("--train-count", type=int, default=2000, help="Number of paired examples to place in train.")
    prepare.add_argument("--test-count", type=int, default=1000, help="Number of paired examples to place in test.")
    prepare.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    prepare.add_argument("--mask-threshold", type=float, default=12.0, help="Per-pixel RGB difference threshold in 0-255 space.")
    prepare.add_argument("--seed", type=int, default=7)

    gen = subparsers.add_parser("generate-masks", help="Generate binary masks from existing clean/ and mask/ folders.")
    gen.add_argument("--output-dir", required=True, help="Output root containing train/ and test/ splits.")
    gen.add_argument("--mask-threshold", type=float, default=12.0, help="Per-pixel RGB difference threshold in 0-255 space.")
    gen.add_argument("--splits", nargs="+", default=["train", "test"], help="Which splits to process.")

    return parser


def _index_unique(paths: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in paths:
        if path.stem in index:
            duplicates.add(path.stem)
        index[path.stem] = path
    if duplicates:
        duplicate_list = ", ".join(sorted(list(duplicates))[:10])
        raise SystemExit(f"Duplicate image stems found: {duplicate_list}")
    return index


def _derive_mask(clean_path: Path, masked_path: Path, threshold: float) -> Image.Image:
    clean = Image.open(clean_path).convert("RGB")
    masked = Image.open(masked_path).convert("RGB")
    if clean.size != masked.size:
        masked = masked.resize(clean.size, resample=Image.Resampling.BILINEAR)

    clean_arr = np.asarray(clean, dtype=np.float32)
    masked_arr = np.asarray(masked, dtype=np.float32)

    # Max across channels: detects changes in any single channel, not just the average.
    # More sensitive than mean for partially-transparent or single-channel composites.
    diff = np.abs(masked_arr - clean_arr).max(axis=2)
    binary = (diff > threshold).astype(np.uint8) * 255
    mask = Image.fromarray(binary, mode="L")

    # Remove isolated noise pixels before expanding (opening: erode then dilate).
    mask = mask.filter(ImageFilter.MinFilter(3))   # erode → kills tiny noise specks
    mask = mask.filter(ImageFilter.MaxFilter(5))   # re-dilate → restore signal size

    # Morphological closing (dilate → erode) fills interior holes in the mask region.
    mask = mask.filter(ImageFilter.MaxFilter(21))  # dilate: bridge gaps, cover edges
    mask = mask.filter(ImageFilter.MinFilter(11))  # erode: pull back to original boundary

    # Soft edge: blur then hard re-binarize for a clean boundary.
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    return mask.point(lambda value: 255 if value > 16 else 0, mode="L")


def _materialize(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def prepare_maskedface_net(
    ffhq_dir: str | Path,
    cmfd_dir: str | Path,
    output_dir: str | Path,
    train_count: int = 2000,
    test_count: int = 1000,
    mode: str = "symlink",
    mask_threshold: float = 12.0,
    seed: int = 7,
) -> dict:
    ffhq_paths = list_images_recursive(ffhq_dir)
    if not ffhq_paths:
        raise SystemExit(f"No FFHQ images found under {ffhq_dir}")
    ffhq_index = _index_unique(ffhq_paths)

    output_root = Path(output_dir)
    masked_paths = list_images_recursive(cmfd_dir)
    matched_pairs: list[tuple[str, Path, Path]] = []
    missing_ffhq = 0

    for masked_path in masked_paths:
        clean_path = ffhq_index.get(masked_path.stem)
        if clean_path is None:
            missing_ffhq += 1
            continue
        matched_pairs.append((masked_path.stem, clean_path, masked_path))

    required = train_count + test_count
    if len(matched_pairs) < required:
            raise SystemExit(
            f"Need at least {required} matched FFHQ/CMFD pairs, found {len(matched_pairs)} under {cmfd_dir}."
        )

    matched_pairs.sort(key=lambda item: item[0])
    rng = random.Random(seed)
    rng.shuffle(matched_pairs)
    selected_pairs = matched_pairs[:required]
    train_pairs = selected_pairs[:train_count]
    test_pairs = selected_pairs[train_count:required]

    def write_split(split_name: str, pairs: list[tuple[str, Path, Path]]) -> None:
        for stem, clean_path, masked_path in pairs:
            split_root = output_root / split_name
            clean_target = split_root / "clean" / f"{stem}{clean_path.suffix.lower()}"
            masked_target = split_root / "masked" / f"{stem}{masked_path.suffix.lower()}"
            mask_target = split_root / "mask" / f"{stem}.png"

            _materialize(clean_path, clean_target, mode)
            _materialize(masked_path, masked_target, mode)
            ensure_dir(mask_target.parent)
            _derive_mask(clean_path, masked_path, mask_threshold).save(mask_target)

    write_split("train", train_pairs)
    write_split("test", test_pairs)

    summary = {
        "matched_available": len(matched_pairs),
        "missing_ffhq": missing_ffhq,
        "train_count": len(train_pairs),
        "test_count": len(test_pairs),
        "cmfd_dir": str(Path(cmfd_dir)),
    }
    save_json(output_root / "manifest.json", summary)
    return summary


def generate_binary_masks(
    output_dir: str | Path,
    mask_threshold: float = 12.0,
    splits: list[str] | None = None,
) -> None:
    """Generate binary masks from paired clean/ and mask/ images into masked/."""
    output_root = Path(output_dir)
    if splits is None:
        splits = ["train", "test"]

    for split in splits:
        clean_dir = output_root / split / "clean"
        cmfd_dir = output_root / split / "mask"
        out_dir = output_root / split / "masked"
        ensure_dir(out_dir)

        # Index clean images by their numeric stem (e.g. "00042")
        clean_index = {p.stem: p for p in clean_dir.iterdir() if p.suffix.lower() == ".png"}

        count = 0
        for cmfd_path in sorted(cmfd_dir.iterdir()):
            if cmfd_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            # CMFD stem is "00042_Mask" → strip suffix to get "00042"
            num = cmfd_path.stem.replace("_Mask", "")
            clean_path = clean_index.get(num)
            if clean_path is None:
                continue
            _derive_mask(clean_path, cmfd_path, mask_threshold).save(out_dir / f"{num}.png")
            count += 1

        print(f"{split}/masked: {count} binary masks generated")


def main() -> None:
    args = build_argparser().parse_args()

    if args.action == "prepare":
        summary = prepare_maskedface_net(
            ffhq_dir=args.ffhq_dir,
            cmfd_dir=args.cmfd_dir,
            output_dir=args.output_dir,
            train_count=args.train_count,
            test_count=args.test_count,
            mode=args.mode,
            mask_threshold=args.mask_threshold,
            seed=args.seed,
        )
        print(summary)

    elif args.action == "generate-masks":
        generate_binary_masks(
            output_dir=args.output_dir,
            mask_threshold=args.mask_threshold,
            splits=args.splits,
        )


if __name__ == "__main__":
    main()
