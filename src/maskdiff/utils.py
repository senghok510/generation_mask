from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_images_recursive(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def stem_index(paths: Iterable[Path]) -> dict[str, Path]:
    return {path.stem: path for path in paths}


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    array = array / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    clipped = tensor.detach().cpu().clamp(-1.0, 1.0)
    array = ((clipped + 1.0) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def tensor_to_mask_image(tensor: torch.Tensor) -> Image.Image:
    clipped = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (clipped.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="L")


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_contact_sheet(clean: Image.Image, mask: Image.Image, generated: Image.Image) -> Image.Image:
    mask_rgb = Image.merge("RGB", (mask, mask, mask))
    width = clean.width + mask_rgb.width + generated.width
    height = max(clean.height, mask_rgb.height, generated.height)
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    x = 0
    for image in (clean, mask_rgb, generated):
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas
