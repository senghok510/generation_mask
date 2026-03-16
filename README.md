# Face Mask Diffusion

A conditional DDPM that synthesizes realistic face masks on clean face images.
The model is conditioned on the original clean face and a binary mask region, so it
edits **only inside the mask** and preserves everything outside it.

Built on top of *Generation of Realistic Facemasked Faces With GANs* (Mumford, 2021),
replacing the global GAN translation with a local, mask-aware diffusion model.

---

## Project Layout

```
generation_mask/
├── src/DDPM/
│   ├── core/                  Shared building blocks
│   │   ├── diffusion.py       DDPM schedule & sampling
│   │   ├── model.py           ConditionalUNet + MaskPredictorUNet
│   │   └── utils.py           Image / filesystem helpers
│   ├── data/                  Data loading & preparation
│   │   ├── dataset.py         FaceMaskDataset (triplet loader)
│   │   ├── maskedface_net.py  FFHQ + CMFD dataset preparation CLI
│   │   └── prepare.py         Incremental dataset builder
│   ├── training/              DDPM training
│   │   └── train.py
│   ├── inference/             DDPM inference
│   │   ├── infer.py           Single-model sampling CLI
│   │   └── pipeline.py        Two-stage pipeline (mask predictor → DDPM)
│   ├── mask_predictor/        U-Net mask predictor (Stage 1)
│   │   ├── train.py
│   │   └── infer.py
│   └── evaluation/            Metrics
│       └── eval.py
├── data/
│   ├── train/
│   │   ├── clean_face/
│   │   ├── masked_face/
│   │   └── binary_mask/
│   ├── validation/
│   │   ├── clean_face/
│   │   ├── masked_face/
│   │   └── binary_mask/
│   └── test/
│       ├── clean_face/
│       ├── masked_face/
│       └── binary_mask/
└── pyproject.toml
```

Files are matched by stem: `clean_face/01000.jpg` pairs with `masked_face/01000.jpg`
and `binary_mask/01000.png`.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Pretrained Checkpoints

Both the **DDPM generator** and the **mask predictor** checkpoints are available here:

> **[Google Drive — checkpoints](https://drive.google.com/drive/folders/1CiMtDYbDme-EXGmAG751ms6LkG0W6_eT?usp=share_link)**

Download and place them anywhere convenient, then pass the paths via `--checkpoint`,
`--ddpm-checkpoint`, or `--mask-checkpoint` as shown below.

---

## Step 0 — Prepare the Dataset

You need [FFHQ](https://github.com/NVlabs/ffhq-dataset) clean faces and
[MaskedFace-Net CMFD](https://github.com/cabani/MaskedFace-Net) masked faces.

```bash
DDPM-prepare \
  --ffhq-dir /path/to/ffhq \
  --cmfd-dir /path/to/MaskedFace-Net/CMFD \
  --output-dir data/ \
  --train-count 2000 \
  --test-count 1000 \
  --mode symlink
```

This creates `data/train/` and `data/test/` with the three subdirectories above,
pairing each CMFD masked face to its corresponding FFHQ clean face by filename stem
and deriving a binary mask from the pixel difference.

---

## Step 1 — Train the Mask Predictor (U-Net, Stage 1)

The mask predictor learns to segment the mask region directly from a clean face image.
It is trained **independently** from the DDPM.

```bash
DDPM-train-mask \
  --train-dir data/train \
  --val-dir   data/validation \
  --save-dir  outputs/mask_predictor \
  --image-size 128 \
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --base-channels 64
```

Checkpoints are written to `outputs/mask_predictor/`:
- `checkpoint_best.pt` — best validation Dice score
- `checkpoint_last.pt` — last epoch

### Use the pretrained mask predictor checkpoint

Download `mask_predictor/checkpoint_best.pt` from the
[Google Drive](https://drive.google.com/drive/folders/1CiMtDYbDme-EXGmAG751ms6LkG0W6_eT?usp=share_link)
and run inference directly:

```bash
DDPM-predict-mask \
  --checkpoint   outputs/mask_predictor/checkpoint_best.pt \
  --input-dir    data/test/clean_face \
  --output-dir   outputs/predicted_masks \
  --image-size   128 \
  --mask-threshold 0.5
```

---

## Step 2 — Train the DDPM Generator

```bash
DDPM-train \
  --train-dir     data/train \
  --val-dir       data/validation \
  --save-dir      outputs/DDPM_base \
  --image-size    128 \
  --batch-size    8 \
  --epochs        50 \
  --timesteps     250 \
  --base-channels 64 \
  --cfg-dropout   0.1 \
  --ema-decay     0.999
```

Checkpoints are written to `outputs/DDPM_base/`:
- `checkpoint_best.pt` — best validation loss
- `checkpoint_last.pt` — last epoch

---

## Inference

### Option A — DDPM only (provide masks manually)

Use this when you already have binary masks for each input image.

```bash
DDPM-generate \
  --checkpoint  outputs/DDPM_base/checkpoint_best.pt \
  --test-dir    data/test \
  --output-dir  outputs/generated \
  --sampler     ddim \
  --sample-steps 100 \
  --guidance-scale 2.5
```

Or with explicit mask directory:

```bash
DDPM-generate \
  --checkpoint outputs/DDPM_base/checkpoint_best.pt \
  --input      data/test/clean_face \
  --mask-dir   data/test/binary_mask \
  --output-dir outputs/generated \
  --sampler    ddim \
  --sample-steps 100 \
  --guidance-scale 2.5
```

### Option B — Two-stage pipeline (mask predictor → DDPM)

Use this when you only have clean face images and want fully automatic mask generation.

```bash
DDPM-pipeline \
  --mask-checkpoint outputs/mask_predictor/checkpoint_best.pt \
  --ddpm-checkpoint outputs/DDPM_base/checkpoint_best.pt \
  --input-dir  data/test/clean_face \
  --output-dir outputs/pipeline_generated \
  --sampler    ddim \
  --sample-steps 100 \
  --guidance-scale 2.5 \
  --save-masks
```

### Using the pretrained checkpoints from Google Drive

Download both checkpoints from the
[Google Drive](https://drive.google.com/drive/folders/1CiMtDYbDme-EXGmAG751ms6LkG0W6_eT?usp=share_link),
place them in `outputs/`, then run Option B above pointing to the downloaded paths.

---

## Evaluation

```bash
DDPM-eval \
  --pred-dir   outputs/generated \
  --target-dir data/test/masked_face \
  --mask-dir   data/test/binary_mask \
  --mode       image \
  --output-json outputs/metrics.json
```

Reported metrics: **MAE** (full / mask region / background), **PSNR**, **FID**, **LPIPS**.

For mask-only evaluation (IoU / Dice):

```bash
DDPM-eval \
  --pred-dir   outputs/predicted_masks \
  --target-dir data/test/binary_mask \
  --mode       mask
```

---

## CLI Reference

| Command | Entry point | Purpose |
|---|---|---|
| `DDPM-prepare` | `DDPM.data.maskedface_net:main` | Pair FFHQ + CMFD, build splits |
| `DDPM-train` | `DDPM.training.train:main` | Train DDPM generator |
| `DDPM-generate` | `DDPM.inference.infer:main` | DDPM inference with provided masks |
| `DDPM-pipeline` | `DDPM.inference.pipeline:main` | Two-stage inference (mask predictor → DDPM) |
| `DDPM-train-mask` | `DDPM.mask_predictor.train:main` | Train U-Net mask predictor |
| `DDPM-predict-mask` | `DDPM.mask_predictor.infer:main` | Run mask predictor on clean images |
| `DDPM-eval` | `DDPM.evaluation.eval:main` | Evaluate predictions |

---

## Notes

- CMFD contains ~67,000 correctly masked synthetic faces aligned to FFHQ.
- MaskedFace-Net is distributed under CC BY-NC-SA 4.0 — research and non-commercial use only.
- The default model is a compact pixel-space DDPM (128×128). For higher quality,
  swap the U-Net backbone into a latent diffusion or inpainting architecture.
- Identity preservation can be improved by adding a frozen face-embedder loss during fine-tuning.
