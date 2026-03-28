# Face Mask Generation with Diffusion Models

This repository contains two diffusion-based implementations for clean-to-masked face synthesis on aligned FFHQ and MaskedFace-Net CMFD pairs:

- `src/DDPM/`: pixel-space conditional DDPM developed by Hok Seng.
- `src/StableDiffusion/`: full-frame Stable Diffusion LoRA pipeline developed by Mohamed Amine Amrani.

Both implementations target the same task introduced in *Generation of Realistic Facemasked Faces With GANs* (Mumford, 2021): generate a realistic masked face from a clean portrait while preserving facial identity and overall image structure.

## Repository Layout

```text
generation_mask/
  src/
    DDPM/
      core/
      evaluation/
      inference/
      mask_predictor/
      training/
    StableDiffusion/
      download_ffhq_subset.py
      facemask_utils.py
      generate_sd_fullframe_results.py
      prepare_paper_dataset.py
      run_pair_metrics.py
      run_pytorch_fid.py
      train_sd_fullframe_lora.py
  data/
  outputs/
  pyproject.toml
  README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset

Both implementations rely on:

- [FFHQ](https://github.com/NVlabs/ffhq-dataset) for clean faces
- [MaskedFace-Net CMFD](https://github.com/cabani/MaskedFace-Net) for masked faces

CMFD is not downloaded by this repository. It must be obtained separately from the official MaskedFace-Net release and placed locally before dataset preparation.

## DDPM Implementation

The DDPM branch is a conditional diffusion pipeline that edits a clean face inside a binary mask region. It includes:

- a conditional DDPM generator in `src/DDPM/training` and `src/DDPM/inference`
- a mask-predictor U-Net in `src/DDPM/mask_predictor`
- DDPM evaluation utilities in `src/DDPM/evaluation`

The package entry points currently exposed for the DDPM workflow are:

| Command | Entry point | Purpose |
|---|---|---|
| `maskdiff-prepare` | `DDPM.data.maskedface_net:main` | Build aligned splits from FFHQ and CMFD |
| `maskdiff-train` | `DDPM.training.train:main` | Train the DDPM generator |
| `maskdiff-generate` | `DDPM.inference.infer:main` | DDPM inference with provided masks |
| `maskdiff-pipeline` | `DDPM.inference.pipeline:main` | Two-stage inference with mask prediction |
| `maskdiff-train-mask` | `DDPM.mask_predictor.train:main` | Train the mask-predictor U-Net |
| `maskdiff-predict-mask` | `DDPM.mask_predictor.infer:main` | Predict mask regions from clean images |
| `maskdiff-eval` | `DDPM.evaluation.eval:main` | Evaluate DDPM outputs |

## Stable Diffusion Implementation

The Stable Diffusion branch is a full-frame conditional diffusion pipeline built on `runwayml/stable-diffusion-inpainting` and adapted with broad LoRA over the UNet.

Method summary:

- backbone: Stable Diffusion inpainting checkpoint
- conditioning: clean face image + text prompt
- target: masked face image
- adaptation: broad LoRA across a large portion of the UNet
- training formulation: full-image conditional diffusion rather than local oracle-mask inpainting

### Stable Diffusion workflow

1. Build the FFHQ subset that matches the available CMFD stems.
2. Prepare aligned paired train/test folders.
3. Train the LoRA-adapted Stable Diffusion model.
4. Generate masked-face predictions.
5. Evaluate FID and paired image metrics.

### 1. Download the matching FFHQ subset

```bash
sd-download-ffhq \
  --cmfd-root /path/to/CMFD \
  --output-dir datasets/FFHQ_subset \
  --train-count 20000 \
  --test-count 2000
```

Equivalent module invocation:

```bash
python -m StableDiffusion.download_ffhq_subset \
  --cmfd-root /path/to/CMFD \
  --output-dir datasets/FFHQ_subset \
  --train-count 20000 \
  --test-count 2000
```

### 2. Prepare the paired dataset

```bash
sd-prepare-dataset \
  --ffhq-root datasets/FFHQ_subset \
  --masked-root /path/to/CMFD \
  --output-root datasets/facemask_aligned \
  --train-count 20000 \
  --test-count 2000 \
  --overwrite \
  --link-mode hardlink
```

Expected layout:

```text
datasets/facemask_aligned/
  paired/
    train/
      clean/
      masked/
    test/
      clean/
      masked/
  manifest.json
```

### 3. Train Stable Diffusion with broad LoRA

```bash
sd-train \
  --data-root datasets/facemask_aligned/paired \
  --output-dir checkpoints/sd_fullframe_broad_lora \
  --resolution 256 \
  --train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --lr-scheduler constant \
  --num-train-epochs 30 \
  --checkpointing-steps 1250 \
  --mixed-precision bf16 \
  --lora-target-mode broad \
  --lora-rank 16 \
  --lora-alpha 16 \
  --gradient-checkpointing \
  --allow-tf32
```

Resumable checkpoints include:

- LoRA weights
- optimizer state
- learning-rate scheduler state
- RNG state
- training metadata

### 4. Generate masked-face predictions

```bash
sd-generate \
  --split-root datasets/facemask_aligned/paired/test \
  --lora-dir checkpoints/sd_fullframe_broad_lora/final \
  --output-dir generated/sd_fullframe_test \
  --num-inference-steps 30 \
  --guidance-scale 7.5
```

### 5. Evaluate outputs

FID:

```bash
sd-eval-fid \
  --real-dir datasets/facemask_aligned/paired/test/masked \
  --fake-dir generated/sd_fullframe_test \
  --device cuda:0
```

Paired metrics:

```bash
sd-eval-pairs \
  --pred-dir generated/sd_fullframe_test \
  --target-dir datasets/facemask_aligned/paired/test/masked \
  --clean-dir datasets/facemask_aligned/paired/test/clean \
  --compute-lpips \
  --device cuda
```

Reported paired metrics:

- MAE over the full image
- MAE inside the inferred mask region
- MAE on the background
- PSNR
- LPIPS

### Stable Diffusion result

- FID on the evaluation split: `17.7`

## Notes

- Dataset files, checkpoints, generated images, and local caches are excluded from version control.
- MaskedFace-Net is distributed under CC BY-NC-SA 4.0 and is intended for research and non-commercial use.
