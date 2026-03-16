from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from DDPM.core.model import MaskPredictorUNet
from DDPM.core.utils import (
    choose_device,
    ensure_dir,
    image_to_tensor,
    list_images,
    mask_to_tensor,
    save_json,
    seed_everything,
    load_mask,
    load_rgb,
    stem_index,
)


class CleanMaskDataset(Dataset):
    """Loads (clean_face, binary_mask) pairs for mask predictor training."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 128,
        train: bool = True,
        clean_subdir: str = "clean_face",
        mask_subdir: str = "binary_mask",
    ) -> None:
        self.image_size = image_size
        self.train = train
        root = Path(root)

        clean_images = list_images(root / clean_subdir)
        if not clean_images:
            raise ValueError(f"No clean images found in {root / clean_subdir}")

        mask_idx = stem_index(list_images(root / mask_subdir))

        self.samples: list[tuple[Path, Path]] = []
        for clean_path in clean_images:
            mask_path = mask_idx.get(clean_path.stem)
            if mask_path is not None:
                self.samples.append((clean_path, mask_path))

        if not self.samples:
            raise ValueError(f"No clean/mask pairs found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        clean_path, mask_path = self.samples[index]
        clean = load_rgb(clean_path)
        mask = load_mask(mask_path)

        if self.train and random.random() < 0.5:
            clean = TF.hflip(clean)
            mask = TF.hflip(mask)

        clean = TF.resize(clean, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        return {
            "name": clean_path.stem,
            "clean": image_to_tensor(clean),
            "mask": mask_to_tensor(mask),
        }


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    return bce_weight * bce + (1.0 - bce_weight) * dice.mean()


def compute_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = (torch.sigmoid(logits) > 0.5).float()
    gt = (target > 0.5).float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    iou = float((intersection / union.clamp(min=1e-6)).item())
    dice = float((2.0 * intersection / (pred.sum() + gt.sum()).clamp(min=1e-6)).item())
    return {"iou": iou, "dice": dice}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net mask predictor (Stage 1).")
    parser.add_argument("--train-dir", required=True, help="Training split directory.")
    parser.add_argument("--val-dir", help="Validation split directory.")
    parser.add_argument("--save-dir", default="outputs/mask_predictor", help="Checkpoint directory.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def build_loader(path: str | None, image_size: int, batch_size: int, workers: int, train: bool) -> DataLoader | None:
    if not path:
        return None
    dataset = CleanMaskDataset(path, image_size=image_size, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def save_checkpoint(
    output_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict[str, float] | None,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "val_metrics": val_metrics,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": {
            "input_channels": 3,
            "base_channels": args.base_channels,
            "dropout": args.dropout,
        },
        "train_config": vars(args),
    }
    torch.save(payload, output_path)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader | None, device: torch.device) -> dict[str, float] | None:
    if loader is None:
        return None
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_loss = 0.0
    count = 0
    for batch in loader:
        clean = batch["clean"].to(device)
        mask = batch["mask"].to(device)
        logits = model(clean)
        loss = bce_dice_loss(logits, mask)
        metrics = compute_metrics(logits, mask)
        total_loss += loss.item()
        total_iou += metrics["iou"]
        total_dice += metrics["dice"]
        count += 1
    n = max(count, 1)
    model.train()
    return {"val_loss": total_loss / n, "iou": total_iou / n, "dice": total_dice / n}


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = ensure_dir(args.save_dir)
    seed_everything(args.seed)

    train_loader = build_loader(args.train_dir, args.image_size, args.batch_size, args.num_workers, train=True)
    val_loader = build_loader(args.val_dir, args.image_size, args.batch_size, args.num_workers, train=False)

    model = MaskPredictorUNet(base_channels=args.base_channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_dice = -1.0

    save_json(output_dir / "config.json", vars(args))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            clean = batch["clean"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(clean)
                loss = bce_dice_loss(logits, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if batch_idx % 20 == 0 or batch_idx == len(train_loader):
                running = total_loss / batch_idx
                print(f"epoch={epoch} batch={batch_idx}/{len(train_loader)} loss={running:.4f}")

        train_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device)

        status = f"epoch={epoch} train_loss={train_loss:.4f}"
        if val_metrics is not None:
            status += f" val_loss={val_metrics['val_loss']:.4f} iou={val_metrics['iou']:.4f} dice={val_metrics['dice']:.4f}"
        print(status)

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, val_metrics, args)
        if val_metrics is None or val_metrics["dice"] > best_dice:
            if val_metrics is not None:
                best_dice = val_metrics["dice"]
            save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, epoch, val_metrics, args)

    print(f"Training finished. Best dice={best_dice:.4f}. Checkpoints saved under {output_dir}")


if __name__ == "__main__":
    main()
