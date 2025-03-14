#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Error: No GPU ID provided."
    echo "Usage: bash run.sh <GPU_ID> (set to -1 to run on CPU only)"
    exit 1  # Exit with an error code
fi

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$1
if [ "$1" -eq -1 ]; then
    echo "CUDA_VISIBLE_DEVICES set to -1, using CPU"
else
    echo "CUDA_VISIBLE_DEVICES set to: $1"
fi

# Get the base directory of this script
BASE_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Run Python script
python "$BASE_DIR/evaluate_mfcc.py"
python "$BASE_DIR/evaluate_mss.py"
python "$BASE_DIR/evaluate_jtfs.py"
python "$BASE_DIR/evaluate_fadtk_models.py"
python "$BASE_DIR/evaluate_music2latent.py"
