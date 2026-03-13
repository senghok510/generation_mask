from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from maskdiff.data import FaceMaskDataset
from maskdiff.diffusion import DiffusionSchedule
from maskdiff.maskedface_net import prepare_maskedface_net
from maskdiff.model import ConditionalUNet


class SmokeTests(unittest.TestCase):
    def test_dataset_and_model_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split = root / "train"
            for name in ("clean", "masked", "mask"):
                (split / name).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), color=(120, 140, 160)).save(split / "clean" / "face1.png")
            Image.new("RGB", (128, 128), color=(100, 120, 140)).save(split / "clean" / "face2.png")
            Image.new("RGB", (128, 128), color=(130, 100, 120)).save(split / "masked" / "face1.png")
            Image.new("RGB", (128, 128), color=(110, 90, 120)).save(split / "masked" / "face2.png")
            Image.new("L", (128, 128), color=0).save(split / "mask" / "face1.png")
            Image.new("L", (128, 128), color=255).save(split / "mask" / "face2.png")

            dataset = FaceMaskDataset(split, image_size=64, train=False)
            sample = dataset[0]
            self.assertEqual(sample["clean"].shape, (3, 64, 64))
            self.assertEqual(sample["masked"].shape, (3, 64, 64))
            self.assertEqual(sample["mask"].shape, (1, 64, 64))

            model = ConditionalUNet(base_channels=16)
            diffusion = DiffusionSchedule(timesteps=8)
            clean = sample["clean"].unsqueeze(0)
            masked = sample["masked"].unsqueeze(0)
            mask = sample["mask"].unsqueeze(0)
            timesteps = torch.randint(0, diffusion.timesteps, (1,), dtype=torch.long)
            noise = torch.randn_like(masked)
            noisy = diffusion.q_sample(masked, timesteps, noise)
            condition = torch.cat([clean, mask], dim=1)
            pred_noise = model(torch.cat([noisy, condition], dim=1), timesteps)
            self.assertEqual(pred_noise.shape, masked.shape)

    def test_maskedface_net_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ffhq_dir = root / "ffhq"
            cmfd_dir = root / "cmfd"
            ffhq_dir.mkdir()
            cmfd_dir.mkdir()

            for idx in range(3):
                stem = f"{idx+1:05d}"
                clean = Image.new("RGB", (64, 64), color=(180, 160, 150))
                clean.save(ffhq_dir / f"{stem}.png")

                masked = Image.new("RGB", (64, 64), color=(180, 160, 150))
                masked.paste((40, 90, 150), (12, 32, 52, 54))
                masked.save(cmfd_dir / f"{stem}.png")

            summary = prepare_maskedface_net(
                ffhq_dir=ffhq_dir,
                cmfd_dir=cmfd_dir,
                output_dir=root / "prepared",
                train_count=2,
                test_count=1,
                mode="copy",
            )
            self.assertEqual(summary["train_count"], 2)
            self.assertEqual(summary["test_count"], 1)

            dataset = FaceMaskDataset(root / "prepared" / "train", image_size=64, train=False)
            self.assertEqual(len(dataset), 2)


if __name__ == "__main__":
    unittest.main()
