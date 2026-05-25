#!/usr/bin/env bash
# =============================================================================
# Grid Search — ViT-B/16 fine-tuned on CIFAR-10
#
# Usage:
#   bash scripts/grid_search_cifar10_vit_b16.sh
#   bash scripts/grid_search_cifar10_vit_b16.sh --clearml_log=True
#
# Notes:
#   - ViT is expensive; start with fewer LR/WD points and small batch sizes.
#   - gradient_accumulation_steps keeps the effective batch at 128.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
OPTIMIZERS=(muonwithadamw)
LEARNING_RATES=(5e-3 1e-2 5e-2)
BATCH_SIZES=(8 32 64)       # цикл по batch_size
WEIGHT_DECAY=0.05             # зафиксированный weight_decay
TRAINABLE_BLOCKS=(4)          # last N transformer blocks to unfreeze
STRATEGIES=(last_n)           # set to (full last_n) to also sweep full fine-tune

CONFIG_FILE="configs/cifar10_vit_b16.py"
MAX_EPOCHS=20
GRAD_ACCUM=2                  # effective batch = batch_size * GRAD_ACCUM

# ---------------------------------------------------------------------------
python data/cifar10/prepare.py

TOTAL=$(( ${#OPTIMIZERS[@]} * ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * \
          ${#TRAINABLE_BLOCKS[@]} * ${#STRATEGIES[@]} ))
RUN=0

echo ""
echo "====================================================================="
echo "  CIFAR-10 ViT-B/16 Grid Search — ${TOTAL} runs total"
echo "  Fixed weight_decay=${WEIGHT_DECAY}"
echo "====================================================================="

for strategy in "${STRATEGIES[@]}"; do
  for blocks in "${TRAINABLE_BLOCKS[@]}"; do
    for opt in "${OPTIMIZERS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
          RUN=$(( RUN + 1 ))
          RUN_NAME="vit-b16-${strategy}-b${blocks}_opt${opt}_lr${lr}_bs${bs}"
          OUT_DIR="out-grid/cifar10-vit-b16/${RUN_NAME}"

          echo ""
          echo "---------------------------------------------------------------------"
          echo "  Run ${RUN}/${TOTAL}: ${RUN_NAME}"
          echo "---------------------------------------------------------------------"

          python train_image.py \
            --config            "${CONFIG_FILE}"   \
            --finetune_strategy "${strategy}"      \
            --trainable_blocks  "${blocks}"        \
            --optimizer_type    "${opt}"           \
            --learning_rate     "${lr}"            \
            --weight_decay      "${WEIGHT_DECAY}"  \
            --batch_size        "${bs}"            \
            --max_epochs        "${MAX_EPOCHS}"    \
            --out_dir           "${OUT_DIR}"       \
            --clearml_task_name "${RUN_NAME}"      \
            --clearml_log=True  \
            "${@}"
        done
      done
    done
  done
done

echo ""
echo "====================================================================="
echo "  All ${TOTAL} CIFAR-10 ViT-B/16 grid runs complete."
echo "  Checkpoints in: out-grid/cifar10-vit-b16/"
echo "====================================================================="