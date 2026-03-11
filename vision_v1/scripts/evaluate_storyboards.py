#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storyboard.evaluation.metrics import evaluate_storyboards
from storyboard.manifests import load_story_manifest
from storyboard.utils.common import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated storyboard outputs.")
    parser.add_argument("--generated-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--scorer-checkpoint", default=None)
    parser.add_argument("--ground-truth-manifest", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_payload = read_json(args.generated_manifest)
    generated_stories = generated_payload["stories"]
    ground_truth_stories = None
    if args.ground_truth_manifest:
        ground_truth_stories = load_story_manifest(args.ground_truth_manifest)
    summary = evaluate_storyboards(
        generated_stories,
        clip_model_name=args.clip_model,
        output_dir=args.output_dir,
        scorer_checkpoint=args.scorer_checkpoint,
        ground_truth_stories=ground_truth_stories,
    )
    print(summary)


if __name__ == "__main__":
    main()
