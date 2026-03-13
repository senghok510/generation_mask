from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from maskdiff.data import FaceMaskDataset
from maskdiff.diffusion import DiffusionSchedule
from maskdiff.model import ConditionalUNet
from maskdiff.utils import choose_device, ensure_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model for face-mask synthesis.")
    parser.add_argument("--train-dir", required=True, help="Training split directory.")
    parser.add_argument("--val-dir", help="Validation split directory.")
    parser.add_argument("--save-dir", default="outputs/maskdiff_base", help="Checkpoint and metadata directory.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cfg-dropout", type=float, default=0.1, help="Condition dropout for classifier-free guidance.")
    parser.add_argument("--mask-weight", type=float, default=4.0, help="Extra loss weight inside the editable mask region.")
    parser.add_argument("--background-weight", type=float, default=2.0, help="Penalty for changing pixels outside the mask.")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, parameter in model_params.items():
            ema_params[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        ema_buffers = dict(ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, buffer in model_buffers.items():
            ema_buffers[name].copy_(buffer)


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float | None,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": {
            "input_channels": 7,
            "output_channels": 3,
            "base_channels": args.base_channels,
            "dropout": args.dropout,
        },
        "train_config": vars(args),
    }
    torch.save(payload, output_path)


def build_loader(path: str | None, image_size: int, batch_size: int, workers: int, train: bool) -> DataLoader | None:
    if not path:
        return None
    dataset = FaceMaskDataset(path, image_size=image_size, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_loss(
    model: nn.Module,
    diffusion: DiffusionSchedule,
    batch: dict[str, torch.Tensor | str],
    cfg_dropout: float,
    mask_weight: float,
    background_weight: float,
    device: torch.device,
) -> torch.Tensor:
    clean = batch["clean"].to(device)
    masked = batch["masked"].to(device)
    mask = batch["mask"].to(device)

    timesteps = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device)
    noise = torch.randn_like(masked)
    noisy = diffusion.q_sample(masked, timesteps, noise)

    condition = torch.cat([clean, mask], dim=1)
    if cfg_dropout > 0.0:
        keep = (torch.rand(clean.shape[0], 1, 1, 1, device=device) > cfg_dropout).float()
        condition = condition * keep

    pred_noise = model(torch.cat([noisy, condition], dim=1), timesteps)
    weights = 1.0 + mask_weight * mask
    noise_loss = ((pred_noise - noise) ** 2 * weights).mean()

    pred_start = diffusion.predict_start_from_noise(noisy, timesteps, pred_noise).clamp(-1.0, 1.0)
    outside = 1.0 - mask
    background_loss = torch.abs((pred_start - clean) * outside).mean()
    return noise_loss + background_weight * background_loss


def evaluate(
    model: nn.Module,
    diffusion: DiffusionSchedule,
    loader: DataLoader | None,
    args: argparse.Namespace,
    device: torch.device,
) -> float | None:
    if loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            loss = compute_loss(
                model=model,
                diffusion=diffusion,
                batch=batch,
                cfg_dropout=0.0,
                mask_weight=args.mask_weight,
                background_weight=args.background_weight,
                device=device,
            )
            total += float(loss.item())
            count += 1
    model.train()
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = ensure_dir(args.save_dir)
    seed_everything(args.seed)

    train_loader = build_loader(args.train_dir, args.image_size, args.batch_size, args.num_workers, train=True)
    val_loader = build_loader(args.val_dir, args.image_size, args.batch_size, args.num_workers, train=False)

    model = ConditionalUNet(base_channels=args.base_channels, dropout=args.dropout).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    diffusion = DiffusionSchedule(timesteps=args.timesteps, device=device)

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val = None

    save_json(output_dir / "config.json", vars(args))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                loss = compute_loss(
                    model=model,
                    diffusion=diffusion,
                    batch=batch,
                    cfg_dropout=args.cfg_dropout,
                    mask_weight=args.mask_weight,
                    background_weight=args.background_weight,
                    device=device,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, args.ema_decay)
            total_loss += float(loss.item())

            if batch_idx % 20 == 0 or batch_idx == len(train_loader):
                running = total_loss / batch_idx
                print(f"epoch={epoch} batch={batch_idx}/{len(train_loader)} loss={running:.4f}")

        train_loss = total_loss / max(len(train_loader), 1)
        val_loss = evaluate(ema_model, diffusion, val_loader, args, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f}"
            + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, ema_model, optimizer, epoch, val_loss, args)
        if val_loss is None:
            save_checkpoint(output_dir / "checkpoint_best.pt", model, ema_model, optimizer, epoch, val_loss, args)
        elif best_val is None or val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "checkpoint_best.pt", model, ema_model, optimizer, epoch, val_loss, args)

    print(f"Training finished. Checkpoints saved under {output_dir}")


if __name__ == "__main__":
    main()
