from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from maskdiff.utils import image_to_tensor, list_images, load_mask, load_rgb, mask_to_tensor, stem_index


@dataclass
class SamplePaths:
    clean: Path
    masked: Path
    mask: Path


def _resolve_clean_dir(root: Path) -> Path:
    clean_dir = root / "clean"
    return clean_dir if clean_dir.exists() else root


class FaceMaskDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 128, train: bool = True) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.train = train

        clean_dir = _resolve_clean_dir(self.root)
        clean_images = list_images(clean_dir)
        if not clean_images:
            raise ValueError(f"No clean images found in {clean_dir}")

        masked_index = stem_index(list_images(self.root / "masked"))
        mask_index = stem_index(list_images(self.root / "mask"))

        self.samples: list[SamplePaths] = []
        for clean_path in clean_images:
            stem = clean_path.stem
            masked_path = masked_index.get(stem)
            mask_path = mask_index.get(stem)
            if masked_path is None or mask_path is None:
                raise ValueError(f"Missing masked or mask file for stem '{stem}' in {self.root}")
            self.samples.append(SamplePaths(clean=clean_path, masked=masked_path, mask=mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def _resize(self, image: Image.Image, is_mask: bool) -> Image.Image:
        mode = InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
        return TF.resize(image, [self.image_size, self.image_size], interpolation=mode)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        clean = load_rgb(sample.clean)
        masked = load_rgb(sample.masked)
        mask = load_mask(sample.mask)

        if self.train and random.random() < 0.5:
            clean = TF.hflip(clean)
            masked = TF.hflip(masked)
            mask = TF.hflip(mask)

        clean = self._resize(clean, is_mask=False)
        masked = self._resize(masked, is_mask=False)
        mask = self._resize(mask, is_mask=True)

        return {
            "name": sample.clean.stem,
            "clean": image_to_tensor(clean),
            "masked": image_to_tensor(masked),
            "mask": mask_to_tensor(mask),
        }
