from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageOps

from storyboard.utils.common import ensure_dir, write_json


def plot_training_history(history: list[dict], output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    target = ensure_dir(output_dir)
    write_json({"history": history}, Path(target) / "history.json")
    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]
    train_accuracy = [record["train_accuracy"] for record in history]
    val_accuracy = [record["val_accuracy"] for record in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Val")
    axes[0].set_title("Ranking Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, train_accuracy, label="Train")
    axes[1].plot(epochs, val_accuracy, label="Val")
    axes[1].set_title("Ranking Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(target) / "training_curves.png", dpi=200)
    plt.close(fig)


def save_storyboard_grid(
    *,
    reference_image_path: str,
    panels: list[dict],
    output_path: str | Path,
    title: str | None = None,
) -> None:
    reference = Image.open(reference_image_path).convert("RGB")
    panel_images = [Image.open(panel["image_path"]).convert("RGB") for panel in panels]
    images = [reference] + panel_images
    cell_width, cell_height = 384, 384
    canvas = Image.new("RGB", (cell_width * len(images), cell_height + 80), color="white")
    for index, image in enumerate(images):
        fitted = ImageOps.fit(image, (cell_width, cell_height))
        canvas.paste(fitted, (index * cell_width, 0))
    output_target = Path(output_path)
    ensure_dir(output_target.parent)
    canvas.save(output_target)
    reference.close()
    for image in panel_images:
        image.close()


def make_gif(frame_paths: list[str], output_path: str | Path, fps: int = 1) -> None:
    ensure_dir(Path(output_path).parent)
    frames = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimsave(output_path, frames, fps=fps)
