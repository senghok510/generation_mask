# Full-Frame Diffusion for Face-Mask Generation

This folder contains an independent implementation of full-frame conditional diffusion for clean-to-masked
face synthesis.

## Method

- Backbone: `runwayml/stable-diffusion-inpainting`
- Adaptation: broad `LoRA`
- Task formulation: full-image conditional diffusion
- Conditioning input: clean face image + text prompt
- Target output: masked face image

The pretrained checkpoint is used as a backbone only. Training and inference are performed on the full image.

## Contents

```text
amine/
  scripts/
    download_ffhq_subset.py
    facemask_utils.py
    generate_sd_fullframe_results.py
    prepare_paper_dataset.py
    run_pair_metrics.py
    run_pytorch_fid.py
    train_sd_fullframe_lora.py
  requirements.txt
  .gitignore
  README.md
```

## Data

The workflow assumes local access to the Correctly Masked Face Dataset (CMFD) from the official MaskedFace-Net
release:

- Repository: [cabani/MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)
- Paper: [MaskedFace-Net - A dataset of correctly/incorrectly masked face images in the context of COVID-19](https://doi.org/10.1016/j.smhl.2020.100144)

The code in this folder does not download CMFD. It only downloads the required FFHQ subset after matching filename
stems against an existing CMFD copy.

## Environment

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the matching FFHQ subset:

```bash
python scripts/download_ffhq_subset.py \
  --cmfd-root /path/to/CMFD \
  --output-dir datasets/FFHQ_subset \
  --train-count 20000 \
  --test-count 2000
```

Prepare the paired train/test dataset:

```bash
python scripts/prepare_paper_dataset.py \
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

## Training

```bash
python scripts/train_sd_fullframe_lora.py \
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

Resumable checkpoints contain:

- LoRA weights
- optimizer state
- learning-rate scheduler state
- RNG state
- training metadata

## Generation

```bash
python scripts/generate_sd_fullframe_results.py \
  --split-root datasets/facemask_aligned/paired/test \
  --lora-dir checkpoints/sd_fullframe_broad_lora/final \
  --output-dir generated/sd_fullframe_test \
  --num-inference-steps 30 \
  --guidance-scale 7.5
```

## Evaluation

FID:

```bash
python scripts/run_pytorch_fid.py \
  --real-dir datasets/facemask_aligned/paired/test/masked \
  --fake-dir generated/sd_fullframe_test
```

Paired metrics:

```bash
python scripts/run_pair_metrics.py \
  --pred-dir generated/sd_fullframe_test \
  --target-dir datasets/facemask_aligned/paired/test/masked \
  --clean-dir datasets/facemask_aligned/paired/test/clean \
  --compute-lpips
```

## Current Result

- FID on the evaluation split: `17.7`

## Notes

- Dataset files, checkpoints, and generated outputs are excluded from version control.