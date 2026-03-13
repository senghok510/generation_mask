# Face Mask Diffusion

This repository is a fresh implementation built on top of the paper *Generation of Realistic Facemasked Faces With GANs* (Samuel Mumford, March 18, 2021). The paper showed that CycleGAN can synthesize masked faces more convincingly than SimGAN, but it also exposed the main weaknesses of that setup:

- the masked-face training set was synthetic and visually limited
- the GAN often changed more than the mouth and nose region
- training quality was sensitive to instability and manual tuning
- the paper explicitly notes that generalization to real masks remains unresolved

This project replaces the global GAN translation setup with a local, conditional diffusion model. The model learns to denoise a masked-face target while conditioning on:

- the original clean face
- a binary mask region that marks where edits are allowed

That change matters because it turns the task from broad style transfer into constrained image editing. In practice, the project is designed to preserve identity and background outside the mask region while allowing the model to synthesize the mask itself.

## What Is Implemented

- a CMFD preparation CLI that pairs FFHQ clean faces with MaskedFace-Net `CMFD`
- a mask-aware conditional DDPM implemented directly in PyTorch
- classifier-free guidance for stronger inference control
- a background preservation loss so the model edits locally instead of rewriting the entire face
- training, generation, and evaluation CLIs

## Why This Improves On The Paper

Compared with the CycleGAN and SimGAN baselines in the paper, this repository makes three concrete improvements:

1. **Local conditioning instead of whole-image translation**
   The model is conditioned on both the clean face and an edit mask. The loss explicitly penalizes changes outside the allowed region.

2. **Diffusion training instead of adversarial training**
   DDPM training is usually easier to stabilize than GAN min-max optimization, especially when the visual change is localized.

3. **A direct path toward realism**
   The pipeline now targets MaskedFace-Net `CMFD` directly instead of relying on a synthetic training fallback.

## Recommended Experiment Path

1. Start with FFHQ clean-face crops and MaskedFace-Net `CMFD`.
2. Build paired clean/masked/mask examples with the preparation CLI.
3. Use the paper-sized split: `2000` train pairs and `1000` test pairs.
4. Train the conditional diffusion model on those paired examples.
5. Evaluate:
   - masked-region MAE
   - background MAE
   - PSNR
   - external identity metrics such as ArcFace cosine similarity if you later add a face-recognition evaluator

## Project Layout

```text
src/maskdiff/
  data.py             Dataset loading and paired data handling
  diffusion.py        DDPM schedule and sampling
  eval.py             Pairwise reconstruction metrics
  infer.py            Sampling CLI
  maskedface_net.py   FFHQ + CMFD preparation CLI
  model.py            Conditional U-Net
  train.py            Training loop
  utils.py            Shared image and filesystem helpers
tests/
  test_smoke.py
```

## Dataset Layout

Training expects one directory per split:

```text
data/train/
  clean/
  masked/
  mask/

data/test/
  clean/
  masked/
  mask/
```

Files are matched by stem, so `clean/0001.jpg` pairs with `masked/0001.png` and `mask/0001.png`.

## Quickstart

Install locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Prepare paired FFHQ + `CMFD` data. By default this selects exactly `2000` train pairs and `1000` test pairs to match the paper.

```bash
maskdiff-prepare \
  --ffhq-dir /path/to/ffhq \
  --cmfd-dir /path/to/MaskedFace-Net/CMFD \
  --output-dir data/mfn_paper \
  --mode symlink
```

Train the diffusion model on the prepared dataset:

```bash
maskdiff-train \
  --train-dir data/mfn_paper/train \
  --val-dir data/mfn_paper/test \
  --image-size 128 \
  --batch-size 8 \
  --epochs 50 \
  --timesteps 250 \
  --save-dir outputs/maskdiff_base
```

Generate masked faces from clean inputs:

```bash
maskdiff-generate \
  --checkpoint outputs/maskdiff_base/checkpoint_best.pt \
  --input data/mfn_paper/test/clean \
  --mask-dir data/mfn_paper/test/mask \
  --output-dir outputs/generated_test
```

Evaluate predictions against paired targets:

```bash
maskdiff-eval \
  --pred-dir outputs/generated_test \
  --target-dir data/mfn_paper/test/masked \
  --mask-dir data/mfn_paper/test/mask
```

## Notes

- CMFD support assumes FFHQ files can be matched by filename stem to the masked samples.
- The preparation CLI samples a deterministic paper-sized subset by default: `2000` train pairs and `1000` test pairs.
- `CMFD` contains `67,193` correctly masked synthetic faces aligned to FFHQ, which is why it matches the paper setup.
- `IMFD` is intentionally excluded from this repo path because the goal is to reproduce the paper’s correctly masked domain before improving on it with diffusion.
- MaskedFace-Net is distributed under `CC BY-NC-SA 4.0`, so this setup is appropriate for research and non-commercial experimentation unless you secure separate rights.
- The default model is a compact pixel-space DDPM so it stays self-contained in this environment.
- If you want the next jump in quality, the most natural extension is to keep the same dataset and losses but swap the U-Net into a latent diffusion or inpainting backbone.
- If your real target is masked-face recognition rather than synthesis, the strongest follow-on experiment is to add an identity loss from a frozen face embedder during fine-tuning.
