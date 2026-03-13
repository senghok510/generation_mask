from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x
        x = self.norm(x).reshape(batch, channels, height * width)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = channels // self.heads

        q = q.reshape(batch, self.heads, head_dim, height * width).transpose(2, 3)
        k = k.reshape(batch, self.heads, head_dim, height * width).transpose(2, 3)
        v = v.reshape(batch, self.heads, head_dim, height * width).transpose(2, 3)

        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(2, 3).reshape(batch, channels, height * width)
        attended = self.proj(attended).reshape(batch, channels, height, width)
        return residual + attended


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Stage(nn.Module):
    def __init__(self, *layers: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_embedding)
            else:
                x = layer(x)
        return x


class ConditionalUNet(nn.Module):
    def __init__(self, input_channels: int = 7, output_channels: int = 3, base_channels: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        time_dim = base_channels * 4

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)

        self.enc1 = Stage(
            ResidualBlock(base_channels, base_channels, time_dim, dropout),
            ResidualBlock(base_channels, base_channels, time_dim, dropout),
        )
        self.down1 = Downsample(base_channels)

        self.enc2 = Stage(
            ResidualBlock(base_channels, base_channels * 2, time_dim, dropout),
            ResidualBlock(base_channels * 2, base_channels * 2, time_dim, dropout),
        )
        self.down2 = Downsample(base_channels * 2)

        self.enc3 = Stage(
            ResidualBlock(base_channels * 2, base_channels * 4, time_dim, dropout),
            AttentionBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4, time_dim, dropout),
        )
        self.down3 = Downsample(base_channels * 4)

        self.mid = Stage(
            ResidualBlock(base_channels * 4, base_channels * 8, time_dim, dropout),
            AttentionBlock(base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 4, time_dim, dropout),
        )

        self.up3 = Upsample(base_channels * 4)
        self.dec3 = Stage(
            ResidualBlock(base_channels * 8, base_channels * 4, time_dim, dropout),
            AttentionBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 2, time_dim, dropout),
        )

        self.up2 = Upsample(base_channels * 2)
        self.dec2 = Stage(
            ResidualBlock(base_channels * 4, base_channels * 2, time_dim, dropout),
            ResidualBlock(base_channels * 2, base_channels, time_dim, dropout),
        )

        self.up1 = Upsample(base_channels)
        self.dec1 = Stage(
            ResidualBlock(base_channels * 2, base_channels, time_dim, dropout),
            ResidualBlock(base_channels, base_channels, time_dim, dropout),
        )

        self.out_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)

        x1 = self.enc1(self.stem(x), time_embedding)
        x2 = self.enc2(self.down1(x1), time_embedding)
        x3 = self.enc3(self.down2(x2), time_embedding)
        x4 = self.mid(self.down3(x3), time_embedding)

        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1), time_embedding)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1), time_embedding)
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1), time_embedding)

        return self.out_conv(self.out_act(self.out_norm(x)))
