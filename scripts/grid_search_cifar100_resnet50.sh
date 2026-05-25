#!/usr/bin/env bash
# =============================================================================
# Grid Search — ResNet-50 fine-tuned on CIFAR-100
#
# Usage:
#   bash scripts/grid_search_cifar100_resnet50.sh
#   bash scripts/grid_search_cifar100_resnet50.sh --clearml_log=True
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
OPTIMIZERS=(adamw sgd)
LEARNING_RATES=(5e-4 1e-4)
WEIGHT_DECAYS=(1e-4 1e-3)
BATCH_SIZES=(64 128)
TRAINABLE_STAGES=(2 3)        # number of ResNet stages to unfreeze

CONFIG_FILE="config/cifar100_resnet50.py"
MAX_EPOCHS=40
FINETUNE_STRATEGY="last_n"

# ---------------------------------------------------------------------------
python data/cifar100/prepare.py

TOTAL=$(( ${#OPTIMIZERS[@]} * ${#LEARNING_RATES[@]} * ${#WEIGHT_DECAYS[@]} * \
          ${#BATCH_SIZES[@]} * ${#TRAINABLE_STAGES[@]} ))
RUN=0

echo ""
echo "====================================================================="
echo "  CIFAR-100 ResNet-50 Grid Search — ${TOTAL} runs total"
echo "====================================================================="

for stages in "${TRAINABLE_STAGES[@]}"; do
  for opt in "${OPTIMIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
          RUN=$(( RUN + 1 ))
          RUN_NAME="resnet50-s${stages}_opt${opt}_lr${lr}_wd${wd}_bs${bs}"
          OUT_DIR="out-grid/cifar100-resnet50/${RUN_NAME}"

          echo ""
          echo "---------------------------------------------------------------------"
          echo "  Run ${RUN}/${TOTAL}: ${RUN_NAME}"
          echo "---------------------------------------------------------------------"

          python train_image.py \
            --config            "${CONFIG_FILE}"       \
            --finetune_strategy "${FINETUNE_STRATEGY}" \
            --trainable_stages  "${stages}"            \
            --optimizer_type    "${opt}"               \
            --learning_rate     "${lr}"                \
            --weight_decay      "${wd}"                \
            --batch_size        "${bs}"                \
            --max_epochs        "${MAX_EPOCHS}"        \
            --out_dir           "${OUT_DIR}"           \
            --clearml_task_name "${RUN_NAME}"          \
            "${@}"
        done
      done
    done
  done
done

echo ""
echo "====================================================================="
echo "  All ${TOTAL} CIFAR-100 ResNet-50 grid runs complete."
echo "  Checkpoints in: out-grid/cifar100-resnet50/"
echo "====================================================================="
