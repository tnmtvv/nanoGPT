#!/bin/bash

# Define the grid search parameters
BATCH_SIZES=(512)
LEARNING_RATES=(4e-2)
OPTIMIZERS=(AdaGramPS_Sqrt)
RANKS=(5)
# ALPHAS=()

# Specify the directory where the config files are located
CONFIG_DIR="configs"

echo "Starting training for all Shakespeare grid search configurations..."

# Loop through each combination
for rank in "${RANKS[@]}"; do
  # for alpha in "${ALPHAS[@]}"; do
    for opt in "${OPTIMIZERS[@]}"; do
      for bs in "${BATCH_SIZES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          # Construct the full path to the config file inside the 'config' directory
          CONFIG_FILE="${CONFIG_DIR}/gpt2_finetune.py"
          # CONFIG_FILE="${CONFIG_DIR}/shakespeare_char_original.py"

          # Check if the config file exists at the specified path
          if [ -f "$CONFIG_FILE" ]; then
            echo "-----------------------------------------------------"
            echo "Running training with optimizer: ${opt}, batch_size: ${bs}, learning_rate: ${lr}, rank: ${rank}"
            echo "-----------------------------------------------------"
            # Execute the main train.py script with the correct config file path
            # torchrun --nproc_per_node=1 train_two_optimizers.py --config "${CONFIG_FILE}" --optimizer "${opt}" --learning_rate_diag "3e-4" --learning_rate_full "${lr}" --batch_size "${bs}" --rank "${rank}" --wandb_run_name "finetune_two_optimiazers_${opt}_bs${bs}_lr${lr}_rank${rank}" --max_iters "5000"
            torchrun --nproc_per_node=1 train.py --config "${CONFIG_FILE}" --optimizer "${opt}" --learning_rate "${lr}" --batch_size "${bs}" --rank "${rank}" --wandb_run_name "finetune_${opt}_bs${bs}_lr${lr}" --max_iters "100"
          else
            echo "Warning: Config file not found, skipping: ${CONFIG_FILE}"
          fi
        done
      done
    done
  # done 
done

echo "----------------------------------------"
echo "All training runs complete."