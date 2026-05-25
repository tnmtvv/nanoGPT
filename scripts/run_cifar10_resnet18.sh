#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fine-tune pretrained ResNet-18 on CIFAR-10
# Run from the repo root:  bash scripts/run_cifar10_resnet18.sh
#
# Optional: enable ClearML logging
#   bash scripts/run_cifar10_resnet18.sh --clearml_log=True
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[*] Preparing CIFAR-10 dataset..."
python data/cifar10/prepare.py

echo "[*] Starting ResNet-18 fine-tuning..."
python train_image.py \
    --config configs/cifar10_resnet18.py \
    "${@}"

echo "[✓] Done. Check out-cifar10-resnet18/ for checkpoints and logs."
