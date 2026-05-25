#!/usr/bin/env bash
# =============================================================================
# Grid Search — TinyViT trained from scratch on CIFAR-10
#
# Usage:
#   bash scripts/grid_search_cifar10_tiny_vit.sh
#   bash scripts/grid_search_cifar10_tiny_vit.sh --clearml_log=True
#
# Notes:
#   - From-scratch ViT is sensitive to LR and warmup. Keep LR modest and
#     increase warmup when the loss diverges early.
#   - The 'base' variant takes significantly longer; start with 'small'.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
VARIANTS=(tiny small)
OPTIMIZERS=(adamw)
LEARNING_RATES=(1e-3 5e-4)
WEIGHT_DECAYS=(0.05 0.1)
DROPOUTS=(0.1 0.2)
WARMUP_EPOCHS_LIST=(5 10)

CONFIG_FILE="config/cifar10_tiny_vit.py"
MAX_EPOCHS=100
BATCH_SIZE=128

# ---------------------------------------------------------------------------
python data/cifar10/prepare.py

TOTAL=$(( ${#VARIANTS[@]} * ${#OPTIMIZERS[@]} * ${#LEARNING_RATES[@]} * \
          ${#WEIGHT_DECAYS[@]} * ${#DROPOUTS[@]} * ${#WARMUP_EPOCHS_LIST[@]} ))
RUN=0

echo ""
echo "====================================================================="
echo "  CIFAR-10 TinyViT Grid Search — ${TOTAL} runs total"
echo "====================================================================="

for variant in "${VARIANTS[@]}"; do
  for warmup in "${WARMUP_EPOCHS_LIST[@]}"; do
    for opt in "${OPTIMIZERS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for wd in "${WEIGHT_DECAYS[@]}"; do
          for dropout in "${DROPOUTS[@]}"; do
            RUN=$(( RUN + 1 ))
            RUN_NAME="tiny-vit-${variant}_opt${opt}_lr${lr}_wd${wd}_do${dropout}_wu${warmup}"
            OUT_DIR="out-grid/cifar10-tiny-vit/${RUN_NAME}"

            echo ""
            echo "---------------------------------------------------------------------"
            echo "  Run ${RUN}/${TOTAL}: ${RUN_NAME}"
            echo "---------------------------------------------------------------------"

            python train_image.py \
              --config            "${CONFIG_FILE}"   \
              --model_variant     "${variant}"       \
              --optimizer_type    "${opt}"           \
              --learning_rate     "${lr}"            \
              --weight_decay      "${wd}"            \
              --dropout           "${dropout}"       \
              --warmup_epochs     "${warmup}"        \
              --batch_size        "${BATCH_SIZE}"    \
              --max_epochs        "${MAX_EPOCHS}"    \
              --out_dir           "${OUT_DIR}"       \
              --clearml_task_name "${RUN_NAME}"      \
              "${@}"
          done
        done
      done
    done
  done
done

echo ""
echo "====================================================================="
echo "  All ${TOTAL} CIFAR-10 TinyViT grid runs complete."
echo "  Checkpoints in: out-grid/cifar10-tiny-vit/"
echo "====================================================================="
