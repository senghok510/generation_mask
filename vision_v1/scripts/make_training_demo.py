#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storyboard.generation import rerank_candidate_bank
from storyboard.utils.common import ensure_dir, read_json
from storyboard.viz import make_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render storyboard selection progress across training epochs.")
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--fps", type=int, default=1)
    return parser.parse_args()


def _checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def main() -> None:
    args = parse_args()
    candidate_bank = read_json(args.candidate_bank)["stories"]
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = ensure_dir(args.output_dir)
    frame_paths = []
    for checkpoint in sorted(checkpoint_dir.glob("epoch_*.pt"), key=_checkpoint_sort_key):
        frame_dir = output_dir / checkpoint.stem
        rerank_candidate_bank(
            candidate_bank,
            scorer_checkpoint=str(checkpoint),
            clip_model_name=args.clip_model,
            beam_size=args.beam_size,
            output_dir=str(frame_dir),
        )
        summary_frame = frame_dir / f"{candidate_bank[0]['story_id']}.png"
        if summary_frame.exists():
            frame_paths.append(str(summary_frame))
    if not frame_paths:
        raise FileNotFoundError("No epoch frames were created. Check your checkpoint directory.")
    make_gif(frame_paths, output_dir / "training_progress.gif", fps=args.fps)
    print(f"Saved demo GIF to {output_dir / 'training_progress.gif'}")


if __name__ == "__main__":
    main()
