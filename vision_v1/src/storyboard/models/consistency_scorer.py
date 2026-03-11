from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ScorerConfig:
    embedding_dim: int
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1


class StoryConsistencyScorer(nn.Module):
    def __init__(self, config: ScorerConfig) -> None:
        super().__init__()
        self.config = config
        fused_dim = config.embedding_dim * 6
        self.panel_fuser = nn.Sequential(
            nn.Linear(fused_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.panel_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.story_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def fuse_panel_features(
        self,
        reference_embeddings: torch.Tensor,
        panel_image_embeddings: torch.Tensor,
        panel_text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        reference_expanded = reference_embeddings.unsqueeze(1).expand_as(panel_image_embeddings)
        fused = torch.cat(
            [
                reference_expanded,
                panel_image_embeddings,
                panel_text_embeddings,
                reference_expanded * panel_image_embeddings,
                panel_image_embeddings * panel_text_embeddings,
                reference_expanded * panel_text_embeddings,
            ],
            dim=-1,
        )
        return self.panel_fuser(fused)

    def forward(
        self,
        reference_embeddings: torch.Tensor,
        panel_image_embeddings: torch.Tensor,
        panel_text_embeddings: torch.Tensor,
        panel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = self.fuse_panel_features(reference_embeddings, panel_image_embeddings, panel_text_embeddings)
        padding_mask = ~panel_mask.bool()
        encoded = self.sequence_encoder(fused, src_key_padding_mask=padding_mask)
        panel_scores = self.panel_head(encoded).squeeze(-1)
        panel_scores = panel_scores.masked_fill(padding_mask, 0.0)
        masked_encoded = encoded * panel_mask.unsqueeze(-1)
        pooled = masked_encoded.sum(dim=1) / panel_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        story_score = self.story_head(pooled).squeeze(-1)
        return {
            "story_score": story_score,
            "panel_scores": panel_scores,
            "pooled_features": pooled,
        }


def save_checkpoint(
    model: StoryConsistencyScorer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    path: str,
) -> None:
    torch.save(
        {
            "config": model.config.__dict__,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[StoryConsistencyScorer, dict]:
    payload = torch.load(path, map_location=map_location)
    config = ScorerConfig(**payload["config"])
    model = StoryConsistencyScorer(config)
    model.load_state_dict(payload["model_state"])
    return model, payload
