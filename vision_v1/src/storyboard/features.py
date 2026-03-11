from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from storyboard.utils.common import chunked, ensure_dir, resolve_device


def _coerce_embedding_tensor(output: object, *, output_name: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    direct_attr = getattr(output, output_name, None)
    if isinstance(direct_attr, torch.Tensor):
        return direct_attr

    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]

    raise TypeError(f"Expected a tensor-like CLIP output, got {type(output).__name__}")


class FeatureBank:
    def __init__(
        self,
        *,
        clip_model_name: str,
        image_features: dict[str, torch.Tensor],
        text_features: dict[str, torch.Tensor],
    ) -> None:
        self.clip_model_name = clip_model_name
        self.image_features = image_features
        self.text_features = text_features
        sample = next(iter(image_features.values()), None)
        if sample is None:
            sample = next(iter(text_features.values()), None)
        self.embedding_dim = int(sample.shape[-1]) if sample is not None else 0

    def get_image(self, image_path: str) -> torch.Tensor:
        return self.image_features[image_path].clone()

    def get_text(self, text: str) -> torch.Tensor:
        return self.text_features[text].clone()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        ensure_dir(target.parent)
        torch.save(
            {
                "clip_model_name": self.clip_model_name,
                "image_features": self.image_features,
                "text_features": self.text_features,
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureBank":
        payload = torch.load(Path(path), map_location="cpu")
        return cls(
            clip_model_name=payload["clip_model_name"],
            image_features=payload["image_features"],
            text_features=payload["text_features"],
        )


def load_clip_components(clip_model_name: str, *, prefer_half: bool = False) -> tuple[CLIPModel, CLIPProcessor, str]:
    device, dtype = resolve_device(prefer_half=prefer_half)
    if device == "cpu":
        dtype = torch.float32
    model = CLIPModel.from_pretrained(clip_model_name)
    model.to(device=device, dtype=dtype)
    model.eval()
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    return model, processor, device


@torch.inference_mode()
def encode_image_paths(
    image_paths: Iterable[str],
    *,
    clip_model_name: str,
    batch_size: int = 32,
    prefer_half: bool = False,
) -> dict[str, torch.Tensor]:
    unique_paths = sorted(set(image_paths))
    model, processor, device = load_clip_components(clip_model_name, prefer_half=prefer_half)
    features: dict[str, torch.Tensor] = {}
    for batch in chunked(unique_paths, batch_size):
        images = [Image.open(path).convert("RGB") for path in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        embeddings = _coerce_embedding_tensor(
            model.get_image_features(**inputs),
            output_name="image_embeds",
        )
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        for path, embedding in zip(batch, embeddings):
            features[path] = embedding
        for image in images:
            image.close()
    return features


@torch.inference_mode()
def encode_texts(
    texts: Iterable[str],
    *,
    clip_model_name: str,
    batch_size: int = 64,
    prefer_half: bool = False,
) -> dict[str, torch.Tensor]:
    unique_texts = sorted(set(texts))
    model, processor, device = load_clip_components(clip_model_name, prefer_half=prefer_half)
    features: dict[str, torch.Tensor] = {}
    for batch in chunked(unique_texts, batch_size):
        inputs = processor(text=list(batch), return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        embeddings = _coerce_embedding_tensor(
            model.get_text_features(**inputs),
            output_name="text_embeds",
        )
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        for text, embedding in zip(batch, embeddings):
            features[text] = embedding
    return features
