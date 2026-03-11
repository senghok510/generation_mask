#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storyboard.manifests import load_story_manifest
from storyboard.utils.common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple story manifests into one file.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stories = []
    for manifest_path in args.manifests:
        stories.extend(load_story_manifest(manifest_path))
    write_json({"num_stories": len(stories), "stories": stories}, args.output_path)
    print(f"Wrote merged manifest with {len(stories)} stories to {args.output_path}")


if __name__ == "__main__":
    main()
