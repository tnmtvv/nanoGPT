#!/usr/bin/env bash
# =============================================================================
# Grid Search — LeNet-5 on MNIST
#
# Usage:
#   bash scripts/grid_search_mnist_cnn.sh
#   bash scripts/grid_search_mnist_cnn.sh --clearml_log=True
#
# All extra flags appended at the bottom of the script are forwarded to
# train_image.py as-is, so you can inject extra overrides from the CLI.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Grid parameters — edit to taste
# ---------------------------------------------------------------------------
OPTIMIZERS=(adamw sgd)
LEARNING_RATES=(1e-3 1e-2)
WEIGHT_DECAYS=(1e-4 1e-3)
BATCH_SIZES=(64 128)
DROPOUTS=(0.0 0.2)

# Fixed settings for this model
CONFIG_FILE="config/mnist_cnn.py"
MAX_EPOCHS=20
DATASET="mnist"
MODEL="lenet5"

# ---------------------------------------------------------------------------
# One-time dataset prep
# ---------------------------------------------------------------------------
echo "[*] Preparing MNIST dataset..."
python data/mnist/prepare.py

# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------
TOTAL=$(( ${#OPTIMIZERS[@]} * ${#LEARNING_RATES[@]} * ${#WEIGHT_DECAYS[@]} * \
          ${#BATCH_SIZES[@]} * ${#DROPOUTS[@]} ))
RUN=0

echo ""
echo "====================================================================="
echo "  MNIST CNN Grid Search — ${TOTAL} runs total"
echo "====================================================================="

for opt in "${OPTIMIZERS[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
      for bs in "${BATCH_SIZES[@]}"; do
        for dropout in "${DROPOUTS[@]}"; do
          RUN=$(( RUN + 1 ))
          RUN_NAME="${MODEL}_opt${opt}_lr${lr}_wd${wd}_bs${bs}_do${dropout}"
          OUT_DIR="out-grid/mnist-cnn/${RUN_NAME}"

          echo ""
          echo "---------------------------------------------------------------------"
          echo "  Run ${RUN}/${TOTAL}: ${RUN_NAME}"
          echo "---------------------------------------------------------------------"

          python train_image.py \
            --config "${CONFIG_FILE}" \
            --optimizer_type  "${opt}"     \
            --learning_rate   "${lr}"      \
            --weight_decay    "${wd}"      \
            --batch_size      "${bs}"      \
            --dropout         "${dropout}" \
            --max_epochs      "${MAX_EPOCHS}" \
            --out_dir         "${OUT_DIR}" \
            --clearml_task_name "${RUN_NAME}" \
            "${@}"    # forward extra CLI flags (e.g. --clearml_log=True)
        done
      done
    done
  done
done

echo ""
echo "====================================================================="
echo "  All ${TOTAL} MNIST CNN grid runs complete."
echo "  Checkpoints in: out-grid/mnist-cnn/"
echo "====================================================================="
