# Face Mask Generation with Diffusion Models

This repository contains two diffusion-based implementations for clean-to-masked face synthesis on aligned FFHQ and MaskedFace-Net CMFD pairs:

- `src/DDPM/`: pixel-space conditional DDPM developed by Hok Seng.
- `src/StableDiffusion/`: full-frame Stable Diffusion LoRA pipeline developed by Mohamed Amine Amrani.

Both implementations target the task introduced in *Generation of Realistic Facemasked Faces With GANs* (Mumford, 2021): generate a realistic masked face from a clean portrait while preserving facial identity and overall image structure.

## Repository Layout

```text
generation_mask/
  src/
    DDPM/
      core/                  Shared building blocks
        diffusion.py         DDPM schedule and sampling
        model.py             ConditionalUNet and MaskPredictorUNet
        utils.py             Image and filesystem helpers
      data/                  Data loading and preparation
        dataset.py           FaceMaskDataset triplet loader
        maskedface_net.py    FFHQ plus CMFD dataset preparation CLI
        prepare.py           Incremental dataset builder
      training/              DDPM training
        train.py
      inference/             DDPM inference
        infer.py             Single-model sampling CLI
        pipeline.py          Two-stage pipeline, mask predictor to DDPM
      mask_predictor/        U-Net mask predictor
        train.py
        infer.py
      evaluation/            DDPM evaluation
        eval.py
    StableDiffusion/
      download_ffhq_subset.py
      facemask_utils.py
      generate_sd_fullframe_results.py
      prepare_paper_dataset.py
      run_pair_metrics.py
      run_pytorch_fid.py
      train_sd_fullframe_lora.py
  data/
    train/
    validation/
    test/
  outputs/
  pyproject.toml
  README.md
```

In the DDPM workflow, files are matched by stem: `clean_face/01000.jpg` pairs with `masked_face/01000.jpg` and `binary_mask/01000.png`.

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

## Pretrained Checkpoints

The DDPM generator and the DDPM mask-predictor checkpoints are available here:

> **[Google Drive - checkpoints](https://drive.google.com/drive/folders/1CiMtDYbDme-EXGmAG751ms6LkG0W6_eT?usp=share_link)**

Download and place them anywhere convenient, then pass the paths via `--checkpoint`, `--ddpm-checkpoint`, or `--mask-checkpoint` as shown below.

## DDPM Implementation

The DDPM branch is a conditional diffusion pipeline that edits a clean face inside a binary mask region and preserves everything outside that region.

Built on top of *Generation of Realistic Facemasked Faces With GANs* (Mumford, 2021), it replaces global GAN translation with a local, mask-aware diffusion model.

### Step 0 - Prepare the DDPM dataset

The DDPM workflow expects clean faces, masked faces, and binary masks:

```text
data/
  train/
    clean_face/
    masked_face/
    binary_mask/
  validation/
    clean_face/
    masked_face/
    binary_mask/
  test/
    clean_face/
    masked_face/
    binary_mask/
```

Prepare the aligned dataset from FFHQ and CMFD:

```bash
maskdiff-prepare \
  --ffhq-dir /path/to/ffhq \
  --cmfd-dir /path/to/MaskedFace-Net/CMFD \
  --output-dir data/ \
  --train-count 2000 \
  --test-count 1000 \
  --mode symlink
```

This builds paired splits by filename stem and derives binary masks from the clean and masked image difference.

### Step 1 - Train the mask predictor

The mask predictor is a U-Net trained independently from the DDPM generator.

```bash
maskdiff-train-mask \
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

- `checkpoint_best.pt`: best validation Dice score
- `checkpoint_last.pt`: last epoch

Use the pretrained mask predictor checkpoint:

```bash
maskdiff-predict-mask \
  --checkpoint   outputs/mask_predictor/checkpoint_best.pt \
  --input-dir    data/test/clean_face \
  --output-dir   outputs/predicted_masks \
  --image-size   128 \
  --mask-threshold 0.5
```

### Step 2 - Train the DDPM generator

```bash
maskdiff-train \
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

- `checkpoint_best.pt`: best validation loss
- `checkpoint_last.pt`: last epoch

### DDPM inference

Option A: DDPM only, with masks provided manually.

```bash
maskdiff-generate \
  --checkpoint  outputs/DDPM_base/checkpoint_best.pt \
  --test-dir    data/test \
  --output-dir  outputs/generated \
  --sampler     ddim \
  --sample-steps 100 \
  --guidance-scale 2.5
```

Or with an explicit mask directory:

```bash
maskdiff-generate \
  --checkpoint outputs/DDPM_base/checkpoint_best.pt \
  --input      data/test/clean_face \
  --mask-dir   data/test/binary_mask \
  --output-dir outputs/generated \
  --sampler    ddim \
  --sample-steps 100 \
  --guidance-scale 2.5
```

Option B: two-stage pipeline, mask predictor then DDPM.

```bash
maskdiff-pipeline \
  --mask-checkpoint outputs/mask_predictor/checkpoint_best.pt \
  --ddpm-checkpoint outputs/DDPM_base/checkpoint_best.pt \
  --input-dir  data/test/clean_face \
  --output-dir outputs/pipeline_generated \
  --sampler    ddim \
  --sample-steps 100 \
  --guidance-scale 2.5 \
  --save-masks
```

Using the pretrained checkpoints from Google Drive:

Download both checkpoints from the linked Google Drive, place them in `outputs/`, then run the two-stage pipeline above with the downloaded paths.

### DDPM evaluation

```bash
maskdiff-eval \
  --pred-dir   outputs/generated \
  --target-dir data/test/masked_face \
  --mask-dir   data/test/binary_mask \
  --mode       image \
  --output-json outputs/metrics.json
```

Reported metrics:

- MAE over the full image
- MAE over the mask region
- MAE over the background
- PSNR
- FID
- LPIPS

Mask-only evaluation:

```bash
maskdiff-eval \
  --pred-dir   outputs/predicted_masks \
  --target-dir data/test/binary_mask \
  --mode       mask
```

### DDPM CLI reference

| Command | Entry point | Purpose |
|---|---|---|
| `maskdiff-prepare` | `DDPM.data.maskedface_net:main` | Pair FFHQ and CMFD, then build DDPM splits |
| `maskdiff-train` | `DDPM.training.train:main` | Train the DDPM generator |
| `maskdiff-generate` | `DDPM.inference.infer:main` | DDPM inference with provided masks |
| `maskdiff-pipeline` | `DDPM.inference.pipeline:main` | Two-stage inference using mask prediction then DDPM |
| `maskdiff-train-mask` | `DDPM.mask_predictor.train:main` | Train the U-Net mask predictor |
| `maskdiff-predict-mask` | `DDPM.mask_predictor.infer:main` | Predict mask regions from clean images |
| `maskdiff-eval` | `DDPM.evaluation.eval:main` | Evaluate DDPM predictions |

## Stable Diffusion Implementation

The Stable Diffusion branch is a full-frame conditional diffusion pipeline built on `runwayml/stable-diffusion-inpainting` and adapted with broad LoRA over the UNet.

Method summary:

- backbone: Stable Diffusion inpainting checkpoint
- conditioning: clean face image plus text prompt
- target: masked face image
- adaptation: broad LoRA across a large portion of the UNet
- training formulation: full-image conditional diffusion rather than local oracle-mask inpainting

### Stable Diffusion workflow

1. Build the FFHQ subset that matches the available CMFD stems.
2. Prepare aligned paired train and test folders.
3. Train the LoRA-adapted Stable Diffusion model.
4. Generate masked-face predictions.
5. Evaluate FID and paired image metrics.

### 1. Download the matching FFHQ subset

```bash
sd-download-ffhq \
  --cmfd-root /path/to/CMFD \
  --output-dir datasets/FFHQ_subset \
  --train-count 20000 \
  --test-count 1000
```

Equivalent module invocation:

```bash
python -m StableDiffusion.download_ffhq_subset \
  --cmfd-root /path/to/CMFD \
  --output-dir datasets/FFHQ_subset \
  --train-count 20000 \
  --test-count 1000
```

### 2. Prepare the paired dataset

```bash
sd-prepare-dataset \
  --ffhq-root datasets/FFHQ_subset \
  --masked-root /path/to/CMFD \
  --output-root datasets/facemask_aligned \
  --train-count 20000 \
  --test-count 1000 \
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

- CMFD contains approximately 67,000 correctly masked synthetic faces aligned to FFHQ.
- MaskedFace-Net is distributed under CC BY-NC-SA 4.0 and is intended for research and non-commercial use.
- The default DDPM model is a compact pixel-space diffusion model at `128x128`.
- For higher quality DDPM outputs, the U-Net backbone can be replaced with a latent diffusion or inpainting architecture.
- Identity preservation can be improved by adding a frozen face-embedder loss during fine-tuning.
- Dataset files, checkpoints, generated images, and local caches are excluded from version control.
