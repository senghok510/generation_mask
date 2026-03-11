from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from storyboard.features import FeatureBank
from storyboard.models.consistency_scorer import StoryConsistencyScorer, save_checkpoint
from storyboard.viz import plot_training_history


class StoryRankingDataset(Dataset):
    def __init__(
        self,
        stories: list[dict],
        feature_bank: FeatureBank,
        *,
        num_panels: int = 4,
        num_negatives: int = 2,
    ) -> None:
        self.stories = stories
        self.feature_bank = feature_bank
        self.num_panels = num_panels
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.stories)

    def _story_to_tensors(self, story: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = self.feature_bank.get_image(story["reference"]["image_path"]).float()
        embedding_dim = self.feature_bank.embedding_dim
        image_embeddings = torch.zeros(self.num_panels, embedding_dim)
        text_embeddings = torch.zeros(self.num_panels, embedding_dim)
        panel_mask = torch.zeros(self.num_panels)
        for panel_index, panel in enumerate(story["panels"][: self.num_panels]):
            image_embeddings[panel_index] = self.feature_bank.get_image(panel["image_path"]).float()
            text_embeddings[panel_index] = self.feature_bank.get_text(panel["caption"]).float()
            panel_mask[panel_index] = 1.0
        return reference, image_embeddings, text_embeddings, panel_mask

    def _sample_negative(self, story_index: int) -> dict:
        anchor = self.stories[story_index]
        negative = {
            "reference": anchor["reference"],
            "panels": [dict(panel) for panel in anchor["panels"][: self.num_panels]],
        }
        strategy = random.choice(("swap_one", "swap_half", "shuffle"))
        if strategy == "shuffle" and len(negative["panels"]) > 1:
            random.shuffle(negative["panels"])
            return negative

        other_story = self.stories[random.randrange(len(self.stories))]
        while other_story["story_id"] == anchor["story_id"]:
            other_story = self.stories[random.randrange(len(self.stories))]
        other_panels = other_story["panels"][: self.num_panels]
        if not other_panels:
            return negative
        if strategy == "swap_one":
            panel_index = random.randrange(min(len(negative["panels"]), len(other_panels)))
            negative["panels"][panel_index]["image_path"] = other_panels[panel_index % len(other_panels)]["image_path"]
        else:
            start_index = max(1, math.floor(len(negative["panels"]) / 2))
            for panel_index in range(start_index, len(negative["panels"])):
                negative["panels"][panel_index]["image_path"] = other_panels[panel_index % len(other_panels)]["image_path"]
        return negative

    def __getitem__(self, index: int) -> dict:
        story = self.stories[index]
        reference, pos_images, pos_texts, panel_mask = self._story_to_tensors(story)
        negative_images = []
        negative_texts = []
        negative_masks = []
        for _ in range(self.num_negatives):
            negative_story = self._sample_negative(index)
            _, neg_images, neg_texts, neg_mask = self._story_to_tensors(negative_story)
            negative_images.append(neg_images)
            negative_texts.append(neg_texts)
            negative_masks.append(neg_mask)
        return {
            "reference_embeddings": reference,
            "positive_image_embeddings": pos_images,
            "positive_text_embeddings": pos_texts,
            "panel_mask": panel_mask,
            "negative_image_embeddings": torch.stack(negative_images),
            "negative_text_embeddings": torch.stack(negative_texts),
            "negative_panel_mask": torch.stack(negative_masks),
        }


@dataclass
class TrainerConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    margin: float
    output_dir: str
    log_every: int = 10


def ranking_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor, margin: float) -> torch.Tensor:
    expanded_positive = positive_scores.unsqueeze(1).expand_as(negative_scores)
    return torch.relu(margin - expanded_positive + negative_scores).mean()


def ranking_accuracy(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    expanded_positive = positive_scores.unsqueeze(1).expand_as(negative_scores)
    return (expanded_positive > negative_scores).float().mean().item()


def _move_batch_to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _score_negatives(model: StoryConsistencyScorer, batch: dict) -> torch.Tensor:
    negative_images = batch["negative_image_embeddings"]
    negative_texts = batch["negative_text_embeddings"]
    negative_masks = batch["negative_panel_mask"]
    num_negatives = negative_images.shape[1]
    scores = []
    for negative_index in range(num_negatives):
        output = model(
            batch["reference_embeddings"],
            negative_images[:, negative_index],
            negative_texts[:, negative_index],
            negative_masks[:, negative_index],
        )
        scores.append(output["story_score"])
    return torch.stack(scores, dim=1)


def evaluate_epoch(
    model: StoryConsistencyScorer,
    dataloader: DataLoader,
    *,
    device: str,
    margin: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            positive_output = model(
                batch["reference_embeddings"],
                batch["positive_image_embeddings"],
                batch["positive_text_embeddings"],
                batch["panel_mask"],
            )
            negative_scores = _score_negatives(model, batch)
            loss = ranking_loss(positive_output["story_score"], negative_scores, margin)
            total_loss += loss.item()
            total_accuracy += ranking_accuracy(positive_output["story_score"], negative_scores)
            total_batches += 1
    return {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": total_accuracy / max(total_batches, 1),
    }


def train_model(
    model: StoryConsistencyScorer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: str,
    config: TrainerConfig,
) -> list[dict]:
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    history: list[dict] = []
    best_accuracy = -1.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                positive_output = model(
                    batch["reference_embeddings"],
                    batch["positive_image_embeddings"],
                    batch["positive_text_embeddings"],
                    batch["panel_mask"],
                )
                negative_scores = _score_negatives(model, batch)
                loss = ranking_loss(positive_output["story_score"], negative_scores, config.margin)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            accuracy = ranking_accuracy(positive_output["story_score"], negative_scores)
            running_loss += loss.item()
            running_accuracy += accuracy
            if step % config.log_every == 0:
                progress.set_postfix(loss=f"{running_loss / step:.4f}", acc=f"{running_accuracy / step:.4f}")

        train_metrics = {
            "loss": running_loss / max(len(train_loader), 1),
            "accuracy": running_accuracy / max(len(train_loader), 1),
        }
        val_metrics = evaluate_epoch(model, val_loader, device=device, margin=config.margin)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)
        plot_training_history(history, output_dir)
        save_checkpoint(model, optimizer, epoch, history, str(checkpoint_dir / f"epoch_{epoch:03d}.pt"))
        save_checkpoint(model, optimizer, epoch, history, str(checkpoint_dir / "last.pt"))
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            save_checkpoint(model, optimizer, epoch, history, str(checkpoint_dir / "best.pt"))
    return history
