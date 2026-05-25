#!/usr/bin/env bash
# =============================================================================
# Grid Search — ConvNet on CIFAR-10 (from scratch)
#
# Usage:
#   bash scripts/grid_search_cifar10_cnn.sh
#   bash scripts/grid_search_cifar10_cnn.sh --clearml_log=True
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
OPTIMIZERS=(adam)
LEARNING_RATES=(1e-3 5e-3 1e-2 5e-2)
WEIGHT_DECAYS=(1e-4)
BATCH_SIZES=(32 64)
VARIANTS=(small)      # ConvNet architecture size
RANKS=(1)

CONFIG_FILE="configs/cifar10_cnn.py"
MAX_EPOCHS=60
DATASET="cifar10"

# ---------------------------------------------------------------------------
python data/cifar10/prepare.py

TOTAL=$(( ${#OPTIMIZERS[@]} * ${#LEARNING_RATES[@]} * ${#WEIGHT_DECAYS[@]} * \
          ${#BATCH_SIZES[@]} * ${#VARIANTS[@]} ))
RUN=0

echo ""
echo "====================================================================="
echo "  CIFAR-10 CNN Grid Search — ${TOTAL} runs total"
echo "====================================================================="

for variant in "${VARIANTS[@]}"; do
  for opt in "${OPTIMIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
          for rank in "${RANKS[@]}"; do
            RUN=$(( RUN + 1 ))
            RUN_NAME="cnn-${variant}_opt${opt}_lr${lr}_wd${wd}_bs${bs}"
            OUT_DIR="out-grid/cifar10-cnn/${RUN_NAME}"

            echo ""
            echo "---------------------------------------------------------------------"
            echo "  Run ${RUN}/${TOTAL}: ${RUN_NAME}"
            echo "---------------------------------------------------------------------"

            python train_image.py \
              --config "${CONFIG_FILE}" \
              --model_variant   "${variant}" \
              --optimizer_type  "${opt}"     \
              --learning_rate   "${lr}"      \
              --weight_decay    "${wd}"      \
              --batch_size      "${bs}"      \
              --max_epochs      "${MAX_EPOCHS}" \
              --out_dir         "${OUT_DIR}" \
              --rank         "${rank}" \
              --clearml_log   "True"\
              --clearml_task_name "${RUN_NAME}" \
              "${@}"
          done
        done
      done
    done
  done
done

echo ""
echo "====================================================================="
echo "  All ${TOTAL} CIFAR-10 CNN grid runs complete."
echo "  Checkpoints in: out-grid/cifar10-cnn/"
echo "====================================================================="
