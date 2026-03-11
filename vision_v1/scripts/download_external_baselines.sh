#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
mkdir -p "${THIRD_PARTY_DIR}"

if [ ! -d "${THIRD_PARTY_DIR}/style-aligned" ]; then
  git clone https://github.com/google/style-aligned.git "${THIRD_PARTY_DIR}/style-aligned"
fi

if [ ! -d "${THIRD_PARTY_DIR}/StoryDiffusion" ]; then
  git clone https://github.com/HVision-NKU/StoryDiffusion.git "${THIRD_PARTY_DIR}/StoryDiffusion"
fi

echo "Downloaded external baselines into ${THIRD_PARTY_DIR}"
