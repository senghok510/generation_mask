from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an aligned paired FFHQ/CMFD dataset in train/test layout."
    )
    parser.add_argument("--ffhq-root", required=True, type=Path, help="Root folder containing FFHQ images.")
    parser.add_argument("--masked-root", required=True, type=Path, help="Root folder containing CMFD masked images.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets") / "facemask_aligned",
        help="Destination root. The script creates paired/train and paired/test inside it.",
    )
    parser.add_argument("--train-count", type=int, default=2000, help="Number of matched pairs to place in train.")
    parser.add_argument("--test-count", type=int, default=1000, help="Number of matched pairs to place in test.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for deterministic pair selection.")
    parser.add_argument(
        "--link-mode",
        choices=("auto", "copy", "symlink", "hardlink"),
        default="auto",
        help="How files are materialized in the prepared dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing prepared dataset root.")
    return parser.parse_args()


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


def canonical_masked_stem(stem: str) -> str:
    return stem[:-5] if stem.endswith("_Mask") else stem


def ensure_clean_output(root: Path, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {root}. Pass --overwrite to replace it.")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def materialize(src: Path, dst: Path, link_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    attempts = ["hardlink", "symlink", "copy"] if link_mode == "auto" else [link_mode]
    for mode in attempts:
        try:
            if mode == "copy":
                shutil.copy2(src, dst)
            elif mode == "symlink":
                dst.symlink_to(src.resolve())
            elif mode == "hardlink":
                os.link(src, dst)
            else:
                raise ValueError(f"Unsupported link mode: {mode}")
            return mode
        except OSError:
            if mode == attempts[-1]:
                raise

    raise RuntimeError("Unable to materialize dataset file.")


def build_pairs(ffhq_root: Path, masked_root: Path) -> tuple[list[tuple[str, Path, Path]], int]:
    ffhq_index = index_unique_by_stem(list_images_recursive(ffhq_root), f"FFHQ under {ffhq_root}")
    masked_paths = list_images_recursive(masked_root)

    matched_pairs: list[tuple[str, Path, Path]] = []
    missing_ffhq = 0
    for masked_path in masked_paths:
        stem = canonical_masked_stem(masked_path.stem)
        clean_path = ffhq_index.get(stem)
        if clean_path is None:
            missing_ffhq += 1
            continue
        matched_pairs.append((stem, clean_path, masked_path.resolve()))

    matched_pairs.sort(key=lambda item: item[0])
    return matched_pairs, missing_ffhq


def select_pairs(
    pairs: list[tuple[str, Path, Path]],
    train_count: int,
    test_count: int,
    seed: int,
) -> tuple[list[tuple[str, Path, Path]], list[tuple[str, Path, Path]]]:
    required = train_count + test_count
    if len(pairs) < required:
        raise ValueError(f"Need at least {required} matched FFHQ/CMFD pairs, found {len(pairs)}.")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[:required]
    return selected[:train_count], selected[train_count:required]


def write_split(
    split_name: str,
    pairs: list[tuple[str, Path, Path]],
    paired_root: Path,
    link_mode: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    clean_dir = paired_root / split_name / "clean"
    masked_dir = paired_root / split_name / "masked"

    for stem, clean_path, masked_path in pairs:
        clean_name = f"{stem}{clean_path.suffix.lower()}"
        masked_name = f"{stem}{masked_path.suffix.lower()}"
        output_clean = clean_dir / clean_name
        output_masked = masked_dir / masked_name

        clean_mode = materialize(clean_path, output_clean, link_mode)
        masked_mode = materialize(masked_path, output_masked, link_mode)

        records.append(
            {
                "stem": stem,
                "split": split_name,
                "clean_src": str(clean_path),
                "masked_src": str(masked_path),
                "paired_clean": str(output_clean.resolve()),
                "paired_masked": str(output_masked.resolve()),
                "materialization": {
                    "paired_clean": clean_mode,
                    "paired_masked": masked_mode,
                },
            }
        )

    return records


def main() -> None:
    args = parse_args()
    matched_pairs, missing_ffhq = build_pairs(args.ffhq_root, args.masked_root)
    train_pairs, test_pairs = select_pairs(matched_pairs, args.train_count, args.test_count, args.seed)

    ensure_clean_output(args.output_root, overwrite=args.overwrite)
    paired_root = args.output_root / "paired"
    paired_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    records.extend(write_split("train", train_pairs, paired_root, args.link_mode))
    records.extend(write_split("test", test_pairs, paired_root, args.link_mode))

    manifest = {
        "selection_logic": "exact_stem_match_sorted_then_seeded_shuffle",
        "ffhq_root": str(args.ffhq_root.resolve()),
        "masked_root": str(args.masked_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "paired_root": str(paired_root.resolve()),
        "matched_available": len(matched_pairs),
        "missing_ffhq": missing_ffhq,
        "train_count": len(train_pairs),
        "test_count": len(test_pairs),
        "seed": args.seed,
        "link_mode": args.link_mode,
        "train_stems": [stem for stem, _, _ in train_pairs],
        "test_stems": [stem for stem, _, _ in test_pairs],
        "records": records,
    }

    manifest_path = args.output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Prepared aligned dataset root: {args.output_root.resolve()}")
    print(f"Paired layout: {paired_root.resolve()}")
    print(f"Matched FFHQ/CMFD pairs available: {len(matched_pairs)}")
    print(f"Train pairs written: {len(train_pairs)}")
    print(f"Test pairs written: {len(test_pairs)}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
