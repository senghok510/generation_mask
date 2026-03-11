from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from storyboard.features import FeatureBank, encode_image_paths, encode_texts
from storyboard.models.consistency_scorer import load_checkpoint
from storyboard.utils.common import write_json


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = torch.nn.functional.normalize(left, dim=-1)
    right = torch.nn.functional.normalize(right, dim=-1)
    return (left * right).sum(dim=-1)


def pairwise_similarity(embeddings: list[torch.Tensor]) -> float:
    if len(embeddings) < 2:
        return 1.0
    similarities = []
    for left, right in itertools.combinations(embeddings, 2):
        similarities.append(cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())
    return float(np.mean(similarities))


def evaluate_storyboards(
    stories: list[dict],
    *,
    clip_model_name: str,
    output_dir: str,
    scorer_checkpoint: str | None = None,
    ground_truth_stories: list[dict] | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = [story["reference_image_path"] for story in stories]
    captions: list[str] = []
    panel_image_paths: list[str] = []
    for story in stories:
        for panel in story["panels"]:
            panel_image_paths.append(panel["image_path"])
            captions.append(panel["caption"])
    image_paths.extend(panel_image_paths)
    for story in ground_truth_stories or []:
        image_paths.append(story["reference"]["image_path"])
        for panel in story["panels"]:
            image_paths.append(panel["image_path"])
    image_features = encode_image_paths(image_paths, clip_model_name=clip_model_name)
    text_features = encode_texts(captions, clip_model_name=clip_model_name)
    feature_bank = FeatureBank(
        clip_model_name=clip_model_name,
        image_features=image_features,
        text_features=text_features,
    )

    scorer = None
    if scorer_checkpoint:
        scorer, _ = load_checkpoint(scorer_checkpoint)
        scorer.eval()

    gt_by_id = {story["story_id"]: story for story in ground_truth_stories or []}
    records = []
    for story in stories:
        reference_embedding = feature_bank.get_image(story["reference_image_path"]).float()
        panel_embeddings = [feature_bank.get_image(panel["image_path"]).float() for panel in story["panels"]]
        text_embeddings = [feature_bank.get_text(panel["caption"]).float() for panel in story["panels"]]
        ref_panel_scores = [
            cosine_similarity(reference_embedding.unsqueeze(0), panel_embedding.unsqueeze(0)).item()
            for panel_embedding in panel_embeddings
        ]
        text_alignment_scores = [
            cosine_similarity(panel_embedding.unsqueeze(0), text_embedding.unsqueeze(0)).item()
            for panel_embedding, text_embedding in zip(panel_embeddings, text_embeddings)
        ]
        row = {
            "story_id": story["story_id"],
            "ref_panel_similarity": float(np.mean(ref_panel_scores)),
            "panel_pair_similarity": pairwise_similarity(panel_embeddings),
            "caption_alignment": float(np.mean(text_alignment_scores)),
        }
        if scorer is not None:
            with torch.inference_mode():
                output = scorer(
                    reference_embedding.unsqueeze(0),
                    torch.stack(panel_embeddings).unsqueeze(0),
                    torch.stack(text_embeddings).unsqueeze(0),
                    torch.ones(1, len(panel_embeddings)),
                )
            row["model_story_score"] = float(output["story_score"].item())
        if story["story_id"] in gt_by_id:
            gt_story = gt_by_id[story["story_id"]]
            gt_embeddings = [
                feature_bank.get_image(panel["image_path"]).float()
                for panel in gt_story["panels"][: len(panel_embeddings)]
            ]
            row["gt_image_similarity"] = float(
                np.mean(
                    [
                        cosine_similarity(panel.unsqueeze(0), gt.unsqueeze(0)).item()
                        for panel, gt in zip(panel_embeddings, gt_embeddings)
                    ]
                )
            )
        records.append(row)

    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_path / "story_metrics.csv", index=False)
    numeric_means = frame.mean(numeric_only=True).to_dict()
    numeric_stds = frame.std(numeric_only=True).to_dict()
    summary = {
        "num_stories": len(records),
        "means": numeric_means,
        "stds": numeric_stds,
        "metrics_csv": str(output_path / "story_metrics.csv"),
    }
    write_json(summary, output_path / "summary.json")
    return summary
