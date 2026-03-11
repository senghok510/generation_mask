# Style-Consistent Storyboard Generation

This repository implements a compact research project for **reference-guided storyboard generation**. The core idea is practical: use a strong pretrained image generator to produce several panel candidates per caption, then train a lightweight **multimodal sequence reranker** that selects the storyboard with the most consistent style relative to a reference frame.

That design gives you:

- a real multimodal research question,
- a trainable component with measurable gains,
- visual results that work well for posters,
- reproducible training, evaluation, and demo rendering.

## Project Framing

**Research question**

Can one reference image enforce a consistent visual style across a multi-panel story better than prompt-only or naive reference-guided generation?

**Hypothesis**

A learned sequence-level reranker can improve style consistency because it reasons jointly over:

- the reference image,
- each generated panel,
- the caption sequence across the full story.

Instead of trusting the first generated sample, the reranker scores candidate sequences and selects the most coherent storyboard.

**Method summary**

1. Download and prepare a story visualization dataset such as `PororoSV`.
2. Use a frozen CLIP encoder to precompute image/text features for the dataset.
3. Train `StoryConsistencyScorer`, a transformer-based reranker over panel embeddings.
4. Generate candidate panels with a pretrained diffusion backbone.
5. Rerank the candidate bank with the trained scorer.
6. Evaluate style consistency, caption alignment, and qualitative story coherence.

## Repository Layout

```text
configs/                     Example configs or notes
docs/                        Report and poster guidance
scripts/                     End-to-end CLI entry points
src/storyboard/              Core package
```

## Recommended Experimental Setup

Dataset:

- Primary: `dhruvrnaik/pororo_storyviz`
- Alternative: Flintstones-style story datasets if you adapt the manifest builder

Generator baselines:

1. `independent`: SDXL without reference image
2. `ip_adapter`: SDXL + IP-Adapter reference image
3. `ours`: SDXL + IP-Adapter + trained reranker

Metrics:

- Reference-to-panel CLIP similarity
- Panel-to-panel CLIP similarity
- Caption alignment
- Model story score
- Optional GT image similarity when paired targets exist

## Installation

Create and activate an environment, then install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Or use the helper script:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

If you want the optional external baselines:

```bash
bash scripts/download_external_baselines.sh
```

## End-to-End Commands

If you prefer one script instead of step-by-step commands:

```bash
source .venv/bin/activate
bash scripts/run_pororo_pipeline.sh
```

### 1. Download PororoSV

```bash
python scripts/download_dataset.py \
  --dataset dhruvrnaik/pororo_storyviz \
  --split train \
  --output-dir data/raw/pororo_storyviz

python scripts/download_dataset.py \
  --dataset dhruvrnaik/pororo_storyviz \
  --split validation \
  --output-dir data/raw/pororo_storyviz
```

If the dataset only exposes `train`, split it manually after manifest generation.

### 2. Build manifests

```bash
python scripts/prepare_story_manifest.py \
  --rows-path data/raw/pororo_storyviz/train_rows.jsonl \
  --output-path data/processed/pororo_storyviz/train_stories.json \
  --dataset-name dhruvrnaik/pororo_storyviz \
  --source-split train \
  --num-future-panels 4

python scripts/prepare_story_manifest.py \
  --rows-path data/raw/pororo_storyviz/validation_rows.jsonl \
  --output-path data/processed/pororo_storyviz/val_stories.json \
  --dataset-name dhruvrnaik/pororo_storyviz \
  --source-split validation \
  --num-future-panels 4
```

### 3. Precompute CLIP features

Merge the manifests first so the feature bank covers every image used in training and validation:

```bash
python scripts/merge_manifests.py \
  --manifests \
    data/processed/pororo_storyviz/train_stories.json \
    data/processed/pororo_storyviz/val_stories.json \
  --output-path data/processed/pororo_storyviz/train_val_stories.json

python scripts/precompute_features.py \
  --manifest data/processed/pororo_storyviz/train_val_stories.json \
  --output-path data/features/pororo_train_val_clip.pt \
  --clip-model openai/clip-vit-large-patch14
```

### 4. Train the reranker

```bash
python scripts/train_reranker.py \
  --train-manifest data/processed/pororo_storyviz/train_stories.json \
  --val-manifest data/processed/pororo_storyviz/val_stories.json \
  --feature-bank data/features/pororo_train_val_clip.pt \
  --output-dir outputs/refstoryrank \
  --epochs 15 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --margin 0.3 \
  --num-negatives 3 \
  --num-panels 4
```

Artifacts:

- `outputs/refstoryrank/checkpoints/best.pt`
- `outputs/refstoryrank/training_curves.png`
- `outputs/refstoryrank/history.json`

### 5. Generate baseline storyboards

Prompt-only baseline:

```bash
python scripts/generate_storyboards.py \
  --manifest data/processed/pororo_storyviz/val_stories.json \
  --output-dir outputs/baseline_independent \
  --backend independent \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --limit 32 \
  --num-candidates 1
```

Reference-guided baseline:

```bash
python scripts/generate_storyboards.py \
  --manifest data/processed/pororo_storyviz/val_stories.json \
  --output-dir outputs/baseline_ip_adapter \
  --backend ip_adapter \
  --use-ip-adapter \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --ip-adapter-repo h94/IP-Adapter \
  --limit 32 \
  --num-candidates 4
```

### 6. Run the reranked method

```bash
python scripts/generate_storyboards.py \
  --manifest data/processed/pororo_storyviz/val_stories.json \
  --output-dir outputs/ours_reranked \
  --backend ip_adapter \
  --use-ip-adapter \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --ip-adapter-repo h94/IP-Adapter \
  --limit 32 \
  --num-candidates 4 \
  --rerank-checkpoint outputs/refstoryrank/checkpoints/best.pt \
  --clip-model openai/clip-vit-large-patch14 \
  --beam-size 4
```

This writes:

- `outputs/ours_reranked/candidate_bank.json`
- `outputs/ours_reranked/selected/selected_storyboards.json`
- `outputs/ours_reranked/selected/*.png`

### 7. Evaluate baselines and ours

```bash
python scripts/evaluate_storyboards.py \
  --generated-manifest outputs/baseline_independent/selected/selected_storyboards.json \
  --output-dir outputs/eval_independent \
  --clip-model openai/clip-vit-large-patch14 \
  --ground-truth-manifest data/processed/pororo_storyviz/val_stories.json

python scripts/evaluate_storyboards.py \
  --generated-manifest outputs/baseline_ip_adapter/selected/selected_storyboards.json \
  --output-dir outputs/eval_ip_adapter \
  --clip-model openai/clip-vit-large-patch14 \
  --ground-truth-manifest data/processed/pororo_storyviz/val_stories.json

python scripts/evaluate_storyboards.py \
  --generated-manifest outputs/ours_reranked/selected/selected_storyboards.json \
  --output-dir outputs/eval_ours \
  --clip-model openai/clip-vit-large-patch14 \
  --scorer-checkpoint outputs/refstoryrank/checkpoints/best.pt \
  --ground-truth-manifest data/processed/pororo_storyviz/val_stories.json
```

### 8. Render an epoch-by-epoch training demo

Use a fixed candidate bank from validation data and rerank it with every epoch checkpoint:

```bash
python scripts/make_training_demo.py \
  --candidate-bank outputs/ours_reranked/candidate_bank.json \
  --checkpoint-dir outputs/refstoryrank/checkpoints \
  --output-dir outputs/refstoryrank/demo \
  --clip-model openai/clip-vit-large-patch14 \
  --beam-size 4 \
  --fps 1
```

This produces:

- `outputs/refstoryrank/demo/training_progress.gif`
- per-epoch selected storyboards in `outputs/refstoryrank/demo/epoch_*/`

## What To Show In The Poster

Poster 1:

- task setup figure
- generator + reranker pipeline
- reranker architecture
- training negatives and ranking loss
- references to StoryDiffusion, StyleAligned, IP-Adapter

Poster 2:

- metric table for 3 methods
- qualitative storyboard grids
- training-progress GIF frames
- failure cases: identity drift, pose collapse, wrong scene object, style leakage

## Suggested Report Claims

Keep the contribution honest and narrow:

- We do **not** train a full diffusion model from scratch.
- We reuse a pretrained generator.
- Our contribution is a lightweight **sequence-level multimodal reranker** for style consistency.
- The key empirical finding is whether reranking improves style coherence without hurting caption alignment.

## Recent Related Work

- [StoryDiffusion](https://arxiv.org/abs/2405.01434)
- [Style Aligned Image Generation via Shared Attention](https://arxiv.org/abs/2312.02133)
- [InstantStyle](https://arxiv.org/abs/2404.02733)
- [IP-Adapter](https://arxiv.org/abs/2308.06721)
- [PororoSV](https://huggingface.co/datasets/dhruvrnaik/pororo_storyviz)

## Notes

- `scripts/download_dataset.py` assumes the dataset uses an `image` column and saves rows to JSONL.
- `scripts/prepare_story_manifest.py` supports datasets with `followings`, or grouped stories if you provide `--story-id-field` and `--panel-index-field`.
- The reranker trains on **precomputed CLIP features**, which is much faster than backpropagating through CLIP.
- For the strongest class submission, run multiple seeds and report mean/std on metrics.
