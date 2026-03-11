#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storyboard.manifests import DEFAULT_ID_CANDIDATES, derive_panel_id_from_followings, infer_field_name
from storyboard.utils.common import ensure_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face story dataset and save it locally.")
    parser.add_argument("--dataset", default="dhruvrnaik/pororo_storyviz")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/raw/pororo_storyviz")
    parser.add_argument("--image-format", default="png")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split)
    output_dir = ensure_dir(args.output_dir)
    image_dir = ensure_dir(output_dir / "images" / args.split)
    rows = []
    sample = dataset[0]
    id_field = infer_field_name(sample, DEFAULT_ID_CANDIDATES)
    for row_index, row in enumerate(dataset):
        if args.limit is not None and row_index >= args.limit:
            break
        image = row["image"].convert("RGB")
        resolved_id = row.get(id_field)
        if resolved_id is None:
            resolved_id = derive_panel_id_from_followings(row)
        resolved_id = str(resolved_id or f"{args.split}_{row_index:06d}")
        image_path = image_dir / f"{resolved_id}.{args.image_format}"
        image.save(image_path)
        metadata = {key: value for key, value in row.items() if key != "image"}
        metadata["id"] = resolved_id
        metadata["image_path"] = str(image_path)
        metadata["_row_index"] = row_index
        rows.append(metadata)
        image.close()
    write_jsonl(rows, output_dir / f"{args.split}_rows.jsonl")
    print(f"Saved {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
