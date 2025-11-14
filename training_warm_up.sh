#!/bin/bash

# Define the grid search parameters
BATCH_SIZES=(128 256 512)
LEARNING_RATES=(1e-5 1e-4 1e-3 1e-2)
OPTIMIZERS=(AdaGram)

# Specify the directory where the config files are located
CONFIG_DIR="config"

echo "Starting training for all Shakespeare grid search configurations..."

# Loop through each combination

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
        CUDA_VISIBLE_DEVICES=4 python train_with_warm_up.py --config "${CONFIG_FILE}" --optimizer "${opt}" --learning_rate_diag "0.001" --learning_rate_full "${lr}" --batch_size_diag "64" --batch_size_full "${bs}" --wandb_run_name "${opt}_bs${bs}_lr${lr}" --max_iters "5000"
      else
        echo "Warning: Config file not found, skipping: ${CONFIG_FILE}"
      fi
    done
  done
done


echo "----------------------------------------"
echo "All training runs complete."