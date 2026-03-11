#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storyboard.features import FeatureBank, encode_image_paths, encode_texts
from storyboard.manifests import collect_assets, load_story_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute CLIP features for a story manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stories = load_story_manifest(args.manifest)
    image_paths, texts = collect_assets(stories)
    feature_bank = FeatureBank(
        clip_model_name=args.clip_model,
        image_features=encode_image_paths(
            image_paths,
            clip_model_name=args.clip_model,
            batch_size=args.image_batch_size,
            prefer_half=True,
        ),
        text_features=encode_texts(
            texts,
            clip_model_name=args.clip_model,
            batch_size=args.text_batch_size,
            prefer_half=True,
        ),
    )
    feature_bank.save(args.output_path)
    print(f"Saved feature bank to {args.output_path}")


if __name__ == "__main__":
    main()
