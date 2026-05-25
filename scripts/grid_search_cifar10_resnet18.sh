#!/usr/bin/env bash
# =============================================================================
# Grid Search — ResNet-18 fine-tuned on CIFAR-10
#
# Usage:
#   bash scripts/grid_search_cifar10_resnet18.sh
#   bash scripts/grid_search_cifar10_resnet18.sh --clearml_log=True
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
OPTIMIZERS=(muonwithadamw AdamW adam)
LEARNING_RATES=(1e-3 5e-3 1e-2 5e-2)
WEIGHT_DECAYS=(1e-4)
BATCH_SIZES=(8 32 64)
VARIANTS=(small)      # ConvNet architecture size
STRATEGIES=(last_n)           # set to (full last_n) to also sweep full fine-tune
RANKS=(1)

CONFIG_FILE="configs/cifar10_resnet18.py"
MAX_EPOCHS=100
TRAINABLE_STAGES=2           # used only when STRATEGY=last_n

# ---------------------------------------------------------------------------
python data/cifar10/prepare.py

TOTAL=$(( ${#OPTIMIZERS[@]} * ${#LEARNING_RATES[@]} * ${#WEIGHT_DECAYS[@]} * \
          ${#BATCH_SIZES[@]} * ${#STRATEGIES[@]} ))
RUN=0

echo ""
echo "====================================================================="
echo "  CIFAR-10 ResNet-18 Grid Search — ${TOTAL} runs total"
echo "====================================================================="

for strategy in "${STRATEGIES[@]}"; do
  for opt in "${OPTIMIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
          for rank in "${RANKS[@]}"; do
            RUN=$(( RUN + 1 ))
            RUN_NAME="resnet18-${strategy}_opt${opt}_lr${lr}_wd${wd}_bs${bs}"
            OUT_DIR="out-grid/cifar10-resnet18/${RUN_NAME}"

            echo ""
            echo "---------------------------------------------------------------------"
            echo "  Run ${RUN}/${TOTAL}: ${RUN_NAME}"
            echo "---------------------------------------------------------------------"

            python train_image.py \
              --config            "${CONFIG_FILE}"    \
              --finetune_strategy "${strategy}"       \
              --trainable_stages  "${TRAINABLE_STAGES}" \
              --optimizer_type    "${opt}"            \
              --learning_rate     "${lr}"             \
              --weight_decay      "${wd}"             \
              --batch_size        "${bs}"             \
              --max_epochs        "${MAX_EPOCHS}"     \
              --out_dir           "${OUT_DIR}"        \
              --clearml_task_name "${RUN_NAME}"       \
              "${@}"
          done
        done
      done
    done
  done
done

echo ""
echo "====================================================================="
echo "  All ${TOTAL} CIFAR-10 ResNet-18 grid runs complete."
echo "  Checkpoints in: out-grid/cifar10-resnet18/"
echo "====================================================================="
