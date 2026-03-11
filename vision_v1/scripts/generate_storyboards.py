#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storyboard.generation import DiffusersStoryboardGenerator, rerank_candidate_bank
from storyboard.manifests import load_story_manifest
from storyboard.utils.common import ensure_dir, read_json, seed_everything, write_json
from storyboard.viz import save_storyboard_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate storyboard panels and optionally rerank them.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--candidate-bank", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["independent", "ip_adapter"], default="ip_adapter")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--use-ip-adapter", action="store_true")
    parser.add_argument("--ip-adapter-repo", default="h94/IP-Adapter")
    parser.add_argument("--ip-adapter-subfolder", default="sdxl_models")
    parser.add_argument("--ip-adapter-weight-name", default="ip-adapter-plus_sdxl_vit-h.safetensors")
    parser.add_argument("--ip-adapter-image-encoder-subfolder", default=None)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.8)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted face, duplicate")
    parser.add_argument("--rerank-checkpoint", default=None)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--beam-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)

    if args.candidate_bank:
        payload = read_json(args.candidate_bank)
        candidate_bank = payload["stories"]
    else:
        if not args.manifest:
            raise ValueError("--manifest is required when generating a new candidate bank")
        stories = load_story_manifest(args.manifest)[: args.limit]
        generator = DiffusersStoryboardGenerator(
            model_id=args.model_id,
            use_ip_adapter=args.use_ip_adapter or args.backend == "ip_adapter",
            ip_adapter_repo=args.ip_adapter_repo,
            ip_adapter_subfolder=args.ip_adapter_subfolder,
            ip_adapter_weight_name=args.ip_adapter_weight_name,
            ip_adapter_image_encoder_subfolder=args.ip_adapter_image_encoder_subfolder,
        )
        candidate_bank = []
        for story in stories:
            bank = generator.generate_candidates(
                story_id=story["story_id"],
                reference_image_path=story["reference"]["image_path"],
                captions=[panel["caption"] for panel in story["panels"]],
                output_dir=str(output_dir / "candidates"),
                num_candidates=args.num_candidates,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                ip_adapter_scale=args.ip_adapter_scale,
            )
            candidate_bank.append(bank)
        write_json({"stories": candidate_bank}, output_dir / "candidate_bank.json")

    if args.rerank_checkpoint:
        selected_stories = rerank_candidate_bank(
            candidate_bank,
            scorer_checkpoint=args.rerank_checkpoint,
            clip_model_name=args.clip_model,
            beam_size=args.beam_size,
            output_dir=str(output_dir / "selected"),
        )
    else:
        selected_stories = []
        for story in candidate_bank:
            selected_story = {
                "story_id": story["story_id"],
                "reference_image_path": story["reference_image_path"],
                "panels": [
                    {"caption": panel["caption"], "image_path": panel["candidate_paths"][0]}
                    for panel in story["panels"]
                ],
            }
            selected_stories.append(selected_story)
            save_storyboard_grid(
                reference_image_path=selected_story["reference_image_path"],
                panels=selected_story["panels"],
                output_path=output_dir / "selected" / f"{selected_story['story_id']}.png",
            )
        write_json({"stories": selected_stories}, output_dir / "selected" / "selected_storyboards.json")

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
