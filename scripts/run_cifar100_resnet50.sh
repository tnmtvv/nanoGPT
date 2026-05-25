#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fine-tune pretrained ResNet-50 on CIFAR-100
# Run from the repo root:  bash scripts/run_cifar100_resnet50.sh
#
# Optional: enable ClearML logging
#   bash scripts/run_cifar100_resnet50.sh --clearml_log=True
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[*] Preparing CIFAR-100 dataset..."
python data/cifar100/prepare.py

echo "[*] Starting ResNet-50 fine-tuning..."
python train_image.py \
    --config configs/cifar100_resnet50.py \
    "${@}"

echo "[✓] Done. Check out-cifar100-resnet50/ for checkpoints and logs."
