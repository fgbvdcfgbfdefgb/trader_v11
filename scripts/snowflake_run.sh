#!/usr/bin/env bash
# Offline training entrypoint tuned for a 4x A10G (23GB) / 48-core / 120GB RAM
# Snowflake container. The repo already contains the dataset - only pip
# packages are fetched; training itself is fully offline.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip
# A10G GPUs: the default PyPI torch wheel is CUDA-enabled on Linux.
python3 -m pip install -r requirements.txt

# Auto-detected layout on 4x A10G:
#   predictor=GPU0, analyzer=GPU1, trader A=GPU2, trader B=GPU3
#   tier L: GRU 768x2 agents, MLP 8192-8192-8192 traders (~147M each),
#   8 days/epoch batched rollouts, 24-day GRU batches, full dataset
#   preloaded into RAM across all cores.
python3 train.py \
  --epochs "${EPOCHS:-2000}" \
  --device "${DEVICE:-auto}" \
  --vram-budget-gb "${VRAM:-20}" \
  --days-per-epoch "${DAYS_PER_EPOCH:-8}" \
  --preload "${PRELOAD:-auto}" \
  --out "${OUT:-runs}"
