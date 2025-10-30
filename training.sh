#!/bin/bash

# Define the grid search parameters
BATCH_SIZES=(1 8 16 32 64)
LEARNING_RATES=(1e-5 1e-4 1e-3 1e-2)
SEEDS=(1 2 3 4 5 6)
OPTIMIZERS=(MuonWithAdamW AdamW AdaGrad AdaGram)

# Specify the directory where the config files are located
CONFIG_DIR="config"

echo "Starting training for all Shakespeare grid search configurations..."

# Loop through each combination
for seed in "${SEEDS[@]}"; do
  for opt in "${OPTIMIZERS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do

        # Format the learning rate to match the filename
        lr_filename=$(printf "%g" "$lr")
        
        # Construct the full path to the config file inside the 'config' directory
        CONFIG_FILE="${CONFIG_DIR}/train_shakespeare_nano.py"

        # Check if the config file exists at the specified path
        if [ -f "$CONFIG_FILE" ]; then
          echo "-----------------------------------------------------"
          echo "Running training with optimizer: ${opt}, batch_size: ${bs}, learning_rate: ${lr}"
          echo "-----------------------------------------------------"
          
          # Execute the main train.py script with the correct config file path
          python train.py --config "${CONFIG_FILE}" --optimizer "${opt}" --learning_rate "${lr}" --batch_size "${bs}" --wandb_run_name "${opt}_bs${bs}_lr${lr_filename}" --max_iters "5000" --seed "${seed}"
        else
          echo "Warning: Config file not found, skipping: ${CONFIG_FILE}"
        fi
      done
    done
  done
done

echo "----------------------------------------"
echo "All training runs complete."