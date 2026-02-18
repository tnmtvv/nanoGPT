#!/usr/bin/env bash
set -euo pipefail

# Inputs
DIRECTORIES=("out-hessian-mini-gpt" "out-hessian-adagram-best")
# Order matches DIRECTORIES: warm_up -> minigpt -> adagram_best
FILES=("hessian_layer_analysis_minigpt.csv" "hessian_layer_analysis_adagram_best.csv")

echo "Starting layer analysis for all runs..."

# Require arrays to be aligned
if [[ ${#DIRECTORIES[@]} -ne ${#FILES[@]} ]]; then
  echo "DIRECTORIES and FILES must have the same length" >&2
  exit 1
fi

# GPU can be overridden: CUDA_DEVICE=0 ./run.sh
CUDA_DEVICE="${CUDA_DEVICE:-3}"

for i in "${!DIRECTORIES[@]}"; do
  dir="${DIRECTORIES[$i]}"
  out="${FILES[$i]}"

  if [[ ! -d "$dir" ]]; then
    echo "Skipping missing directory: $dir" >&2
    continue
  fi

  echo "-----------------------------------------------------"
  echo "Analyzing checkpoint: ${dir}"
  echo "Writing to: ${out}"
  echo "-----------------------------------------------------"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python layer_analysis.py \
    --checkpoint_dir "${dir}" \
    --output_csv "${out}"
done

echo "----------------------------------------"
echo "All analyses complete."
