#!/usr/bin/env bash
# =============================================================================
# Grid Search — ViT-B/16 fine-tuned on CIFAR-100
#
# Usage:
#   bash scripts/grid_search_cifar100_vit_b16.sh
#   bash scripts/grid_search_cifar100_vit_b16.sh --clearml_log=True
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
OPTIMIZERS=(adamw)
LEARNING_RATES=(5e-5 1e-5)
WEIGHT_DECAYS=(0.1 0.3)
DROPOUTS=(0.0 0.1)
MAX_EPOCHS_LIST=(20 30)
# BATCH_SIZES=(8 32 64)


CONFIG_FILE="config/cifar100_vit_b16.py"
BATCH_SIZE=64
GRAD_ACCUM=2

# ---------------------------------------------------------------------------
python data/cifar100/prepare.py

TOTAL=$(( ${#OPTIMIZERS[@]} * ${#LEARNING_RATES[@]} * ${#WEIGHT_DECAYS[@]} * \
          ${#DROPOUTS[@]} * ${#MAX_EPOCHS_LIST[@]} ))
RUN=0

echo ""
echo "====================================================================="
echo "  CIFAR-100 ViT-B/16 Grid Search — ${TOTAL} runs total"
echo "====================================================================="

for epochs in "${MAX_EPOCHS_LIST[@]}"; do
  for opt in "${OPTIMIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        for dropout in "${DROPOUTS[@]}"; do
          RUN=$(( RUN + 1 ))
          RUN_NAME="vit-b16-c100_opt${opt}_lr${lr}_wd${wd}_do${dropout}_ep${epochs}"
          OUT_DIR="out-grid/cifar100-vit-b16/${RUN_NAME}"

          echo ""
          echo "---------------------------------------------------------------------"
          echo "  Run ${RUN}/${TOTAL}: ${RUN_NAME}"
          echo "---------------------------------------------------------------------"

          python train_image.py \
            --config            "${CONFIG_FILE}"   \
            --optimizer_type    "${opt}"           \
            --learning_rate     "${lr}"            \
            --weight_decay      "${wd}"            \
            --dropout           "${dropout}"       \
            --batch_size        "${BATCH_SIZE}"    \
            --max_epochs        "${epochs}"        \
            --out_dir           "${OUT_DIR}"       \
            --clearml_task_name "${RUN_NAME}"      \
            "${@}"
        done
      done
    done
  done
done

echo ""
echo "====================================================================="
echo "  All ${TOTAL} CIFAR-100 ViT-B/16 grid runs complete."
echo "  Checkpoints in: out-grid/cifar100-vit-b16/"
echo "====================================================================="
