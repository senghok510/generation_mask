#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from storyboard.features import FeatureBank
from storyboard.manifests import load_story_manifest
from storyboard.models.consistency_scorer import ScorerConfig, StoryConsistencyScorer
from storyboard.training.trainer import StoryRankingDataset, TrainerConfig, train_model
from storyboard.utils.common import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the style consistency reranker.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-panels", type=int, default=4)
    parser.add_argument("--num-negatives", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    train_stories = load_story_manifest(args.train_manifest)
    val_stories = load_story_manifest(args.val_manifest)
    feature_bank = FeatureBank.load(args.feature_bank)
    train_dataset = StoryRankingDataset(
        train_stories,
        feature_bank,
        num_panels=args.num_panels,
        num_negatives=args.num_negatives,
    )
    val_dataset = StoryRankingDataset(
        val_stories,
        feature_bank,
        num_panels=args.num_panels,
        num_negatives=args.num_negatives,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device, _ = resolve_device(prefer_half=False)
    model = StoryConsistencyScorer(
        ScorerConfig(
            embedding_dim=feature_bank.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    ).to(device)
    history = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        config=TrainerConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            margin=args.margin,
            output_dir=args.output_dir,
            log_every=args.log_every,
        ),
    )
    best_checkpoint = Path(args.output_dir) / "checkpoints" / "best.pt"
    print(f"Training complete. Best checkpoint: {best_checkpoint}")
    print(f"Final epoch record: {history[-1]}")


if __name__ == "__main__":
    main()
