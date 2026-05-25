#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train ConvNet (medium) on CIFAR-10 from scratch
# Run from the repo root:  bash scripts/run_cifar10_cnn.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[*] Preparing CIFAR-10 dataset..."
python data/cifar10/prepare.py

echo "[*] Starting training..."
python train_image.py \
    --config configs/cifar10_cnn.py \
    "${@}"

echo "[✓] Done. Check out-cifar10-cnn-medium/ for checkpoints and logs."
