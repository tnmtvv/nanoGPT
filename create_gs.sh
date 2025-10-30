#!/bin/bash

# Define the grid search parameters
BATCH_SIZES=(1 8 16 32 64)
LEARNING_RATES=(1e-5 1e-4 1e-3 1e-2)

# The name of your template configuration file
TEMPLATE_FILE="train_shakespeare_char.py"

# Check if the template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file '$TEMPLATE_FILE' not found."
    exit 1
fi

echo "Generating training files for Shakespeare grid search..."

# Loop through each combination of batch size and learning rate
for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    # Format the learning rate to avoid issues with scientific notation in filenames
    lr_filename=$(printf "%g" "$lr")
    
    # Define a unique filename for the new script
    NEW_FILENAME="train_shakespeare_char_bs${bs}_lr${lr_filename}.py"
    
    # Define a unique WandB run name and output directory
    RUN_NAME="sc-bs${bs}-lr${lr_filename}"
    OUT_DIR="out-${RUN_NAME}"

    echo "Creating ${NEW_FILENAME}"

    # Use sed to replace the default parameters in the template file.
    # Note how we match the existing lines in your config file.
    sed \
      -e "s/wandb_run_name = 'mini-gpt-adagram'/wandb_run_name = '${RUN_NAME}'/" \
      -e "s/out_dir = 'out-shakespeare-char'/out_dir = '${OUT_DIR}'/" \
      -e "s/batch_size = 64/batch_size = ${bs}/" \
      -e "s/learning_rate = 1e-3/learning_rate = ${lr}/" \
      "$TEMPLATE_FILE" > "$NEW_FILENAME"
  done
done

echo "----------------------------------------"
echo "All training files created successfully."
echo "Each file is a config that can be run with your main train.py script."
