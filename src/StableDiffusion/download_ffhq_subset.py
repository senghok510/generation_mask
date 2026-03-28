from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the FFHQ images needed for the aligned clean-to-masked "
            "face synthesis experiment."
        )
    )
    parser.add_argument("--cmfd-root", required=True, type=Path, help="Root folder containing CMFD images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to download the selected FFHQ files.")
    parser.add_argument("--train-count", type=int, default=2000, help="Number of train stems to select.")
    parser.add_argument("--test-count", type=int, default=1000, help="Number of test stems to select.")
    parser.add_argument("--seed", type=int, default=7, help="Selection seed.")
    parser.add_argument("--repo-id", default="marcosv/ffhq-dataset", help="Hugging Face dataset mirror to use.")
    parser.add_argument("--dry-run", action="store_true", help="Only compute and write the manifest without downloading.")
    return parser.parse_args()


def list_images_recursive(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def canonical_stem(stem: str) -> str:
    return stem[:-5] if stem.endswith("_Mask") else stem


def collect_cmfd_stems(cmfd_root: Path) -> list[str]:
    stems = []
    for path in list_images_recursive(cmfd_root):
        stem = canonical_stem(path.stem)
        if stem.isdigit():
            stems.append(stem)
    unique = sorted(set(stems))
    if not unique:
        raise ValueError(f"No numeric CMFD stems found under {cmfd_root}")
    return unique


def select_stems(
    stems: list[str],
    train_count: int,
    test_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    required = train_count + test_count
    if len(stems) < required:
        raise ValueError(f"Need at least {required} CMFD stems, found {len(stems)}")

    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[:required]
    return selected[:train_count], selected[train_count:required]


def ffhq_allow_pattern(stem: str) -> str:
    numeric = int(stem)
    part = numeric // 10000 + 1
    return f"Part{part}/{stem}.png"


def main() -> None:
    args = parse_args()

    stems = collect_cmfd_stems(args.cmfd_root)
    train_stems, test_stems = select_stems(
        stems,
        args.train_count,
        args.test_count,
        args.seed,
    )
    selected = train_stems + test_stems
    patterns = [ffhq_allow_pattern(stem) for stem in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cmfd_root": str(args.cmfd_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "repo_id": args.repo_id,
        "seed": args.seed,
        "train_count": len(train_stems),
        "test_count": len(test_stems),
        "train_stems": train_stems,
        "test_stems": test_stems,
        "allow_patterns": patterns,
    }
    manifest_path = args.output_dir / "ffhq_subset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Selected {len(selected)} FFHQ stems from CMFD")
    print(f"Manifest written to {manifest_path.resolve()}")

    if args.dry_run:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed in the active interpreter. "
            "Install it in the active environment, then rerun this command."
        ) from exc

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.output_dir),
        allow_patterns=patterns,
    )

    print(f"Downloaded {len(selected)} FFHQ images into {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
