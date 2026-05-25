#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train LeNet-5 on MNIST
# Run from the repo root:  bash scripts/run_mnist_cnn.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Prepare dataset (idempotent — skips download if already cached) ──────
echo "[*] Preparing MNIST dataset..."
python data/mnist/prepare.py

# ── 2. Train ─────────────────────────────────────────────────────────────────
echo "[*] Starting training..."
python train_image.py \
    --config config/mnist_cnn.py \
    "${@}"   # forward any extra CLI flags, e.g. --clearml_log=True

echo "[✓] Done. Check out-mnist-lenet5/ for checkpoints and logs."
