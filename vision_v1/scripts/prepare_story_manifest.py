#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storyboard.manifests import build_story_manifest_from_rows, save_story_manifest
from storyboard.utils.common import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert downloaded rows into a storyboard manifest.")
    parser.add_argument("--rows-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--num-future-panels", type=int, default=4)
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--caption-field", default=None)
    parser.add_argument("--image-field", default="image_path")
    parser.add_argument("--followings-field", default="followings")
    parser.add_argument("--story-id-field", default=None)
    parser.add_argument("--panel-index-field", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.rows_path)
    stories = build_story_manifest_from_rows(
        rows,
        dataset_name=args.dataset_name,
        source_split=args.source_split,
        num_future_panels=args.num_future_panels,
        id_field=args.id_field,
        caption_field=args.caption_field,
        image_field=args.image_field,
        followings_field=args.followings_field,
        story_id_field=args.story_id_field,
        panel_index_field=args.panel_index_field,
    )
    save_story_manifest(
        stories,
        args.output_path,
        dataset_name=args.dataset_name,
        source_split=args.source_split,
        extra={"rows_path": args.rows_path},
    )
    print(f"Wrote {len(stories)} stories to {args.output_path}")


if __name__ == "__main__":
    main()
