#!/bin/bash
set -euo pipefail

CONFIG_DIR="config"
CONFIG_FILE="${CONFIG_DIR}/train_shakespeare_char.py"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file not found: ${CONFIG_FILE}"
  exit 1
fi

CUDA_VISIBLE_DEVICES=7
MAX_ITERS=5000

run_one () {
  local opt="$1"
  local bs="$2"
  local lr="$3"
  local rank="${4:-}"
  local out_dir="$5"

  echo "-----------------------------------------------------"
  if [ -n "$rank" ]; then
    echo "Running: opt=${opt}, bs=${bs}, lr=${lr}, rank=${rank}, out_dir=${out_dir}"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
      python train_with_chpts.py --config "${CONFIG_FILE}" \
        --optimizer "${opt}" \
        --learning_rate "${lr}" \
        --batch_size "${bs}" \
        --rank "${rank}" \
        --out_dir "${out_dir}" \
        --wandb_run_name "${opt}_bs${bs}_lr${lr}_rank${rank}" \
        --max_iters "${MAX_ITERS}"
  else
    echo "Running: opt=${opt}, bs=${bs}, lr=${lr}, out_dir=${out_dir}"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
      python train_with_chpts.py --config "${CONFIG_FILE}" \
        --optimizer "${opt}" \
        --learning_rate "${lr}" \
        --batch_size "${bs}" \
        --out_dir "${out_dir}" \
        --wandb_run_name "${opt}_bs${bs}_lr${lr}" \
        --max_iters "${MAX_ITERS}"
  fi
  echo "-----------------------------------------------------"
}

echo "Starting selected training runs..."

# # 1) AdamW with bs=16 lr=1e-4
# run_one "AdamW" 16 1e-4 "" "out_adamw_bs16_lr1e-4"

# # 2) MuonWithAdamW with bs=16 lr=1e-4
run_one "MuonWithAdamW" 16 1e-3 "" "out_muonwithadamw_bs16_lr1e-3"

# 3) AdaGram (you wrote AdamGram; in your original OPTIMIZERS list it's AdaGramPS)
#    bs=512 lr=1e-2 ranks in {1,3,5}
# for rank in 1 3 5; do
#   run_one "AdaGram" 512 1e-2 "${rank}" "out_adagramps_bs512_lr1e-2_rank${rank}"
# done

echo "All requested runs complete."
