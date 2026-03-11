#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${ROOT_DIR}"

export MPLCONFIGDIR="${ROOT_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

DATASET_NAME="${DATASET_NAME:-dhruvrnaik/pororo_storyviz}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14}"
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
IP_ADAPTER_REPO="${IP_ADAPTER_REPO:-h94/IP-Adapter}"
LIMIT="${LIMIT:-32}"
NUM_CANDIDATES="${NUM_CANDIDATES:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

python scripts/download_dataset.py \
  --dataset "${DATASET_NAME}" \
  --split train \
  --output-dir data/raw/pororo_storyviz

python scripts/download_dataset.py \
  --dataset "${DATASET_NAME}" \
  --split validation \
  --output-dir data/raw/pororo_storyviz

python scripts/prepare_story_manifest.py \
  --rows-path data/raw/pororo_storyviz/train_rows.jsonl \
  --output-path data/processed/pororo_storyviz/train_stories.json \
  --dataset-name "${DATASET_NAME}" \
  --source-split train \
  --num-future-panels 4

python scripts/prepare_story_manifest.py \
  --rows-path data/raw/pororo_storyviz/validation_rows.jsonl \
  --output-path data/processed/pororo_storyviz/val_stories.json \
  --dataset-name "${DATASET_NAME}" \
  --source-split validation \
  --num-future-panels 4

python scripts/merge_manifests.py \
  --manifests \
    data/processed/pororo_storyviz/train_stories.json \
    data/processed/pororo_storyviz/val_stories.json \
  --output-path data/processed/pororo_storyviz/train_val_stories.json

python scripts/precompute_features.py \
  --manifest data/processed/pororo_storyviz/train_val_stories.json \
  --output-path data/features/pororo_train_val_clip.pt \
  --clip-model "${CLIP_MODEL}"

python scripts/train_reranker.py \
  --train-manifest data/processed/pororo_storyviz/train_stories.json \
  --val-manifest data/processed/pororo_storyviz/val_stories.json \
  --feature-bank data/features/pororo_train_val_clip.pt \
  --output-dir "${OUTPUT_ROOT}/refstoryrank" \
  --epochs 15 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --margin 0.3 \
  --num-negatives 3 \
  --num-panels 4

python scripts/generate_storyboards.py \
  --manifest data/processed/pororo_storyviz/val_stories.json \
  --output-dir "${OUTPUT_ROOT}/baseline_independent" \
  --backend independent \
  --model-id "${MODEL_ID}" \
  --limit "${LIMIT}" \
  --num-candidates 1

python scripts/generate_storyboards.py \
  --manifest data/processed/pororo_storyviz/val_stories.json \
  --output-dir "${OUTPUT_ROOT}/baseline_ip_adapter" \
  --backend ip_adapter \
  --use-ip-adapter \
  --model-id "${MODEL_ID}" \
  --ip-adapter-repo "${IP_ADAPTER_REPO}" \
  --limit "${LIMIT}" \
  --num-candidates "${NUM_CANDIDATES}"

python scripts/generate_storyboards.py \
  --manifest data/processed/pororo_storyviz/val_stories.json \
  --output-dir "${OUTPUT_ROOT}/ours_reranked" \
  --backend ip_adapter \
  --use-ip-adapter \
  --model-id "${MODEL_ID}" \
  --ip-adapter-repo "${IP_ADAPTER_REPO}" \
  --limit "${LIMIT}" \
  --num-candidates "${NUM_CANDIDATES}" \
  --rerank-checkpoint "${OUTPUT_ROOT}/refstoryrank/checkpoints/best.pt" \
  --clip-model "${CLIP_MODEL}" \
  --beam-size 4

python scripts/evaluate_storyboards.py \
  --generated-manifest "${OUTPUT_ROOT}/baseline_independent/selected/selected_storyboards.json" \
  --output-dir "${OUTPUT_ROOT}/eval_independent" \
  --clip-model "${CLIP_MODEL}" \
  --ground-truth-manifest data/processed/pororo_storyviz/val_stories.json

python scripts/evaluate_storyboards.py \
  --generated-manifest "${OUTPUT_ROOT}/baseline_ip_adapter/selected/selected_storyboards.json" \
  --output-dir "${OUTPUT_ROOT}/eval_ip_adapter" \
  --clip-model "${CLIP_MODEL}" \
  --ground-truth-manifest data/processed/pororo_storyviz/val_stories.json

python scripts/evaluate_storyboards.py \
  --generated-manifest "${OUTPUT_ROOT}/ours_reranked/selected/selected_storyboards.json" \
  --output-dir "${OUTPUT_ROOT}/eval_ours" \
  --clip-model "${CLIP_MODEL}" \
  --scorer-checkpoint "${OUTPUT_ROOT}/refstoryrank/checkpoints/best.pt" \
  --ground-truth-manifest data/processed/pororo_storyviz/val_stories.json

python scripts/make_training_demo.py \
  --candidate-bank "${OUTPUT_ROOT}/ours_reranked/candidate_bank.json" \
  --checkpoint-dir "${OUTPUT_ROOT}/refstoryrank/checkpoints" \
  --output-dir "${OUTPUT_ROOT}/refstoryrank/demo" \
  --clip-model "${CLIP_MODEL}" \
  --beam-size 4 \
  --fps 1

echo "Pipeline finished. Outputs are under ${OUTPUT_ROOT}"
