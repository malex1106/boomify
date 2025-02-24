#!/bin/bash

# Define the paths to your datasets and corresponding output files
data_paths=(
    "/path/to/dataset1"
    "/path/to/dataset2"
    "/path/to/dataset3"
)

output_paths=(
    "/path/to/output1.npz"
    "/path/to/output2.npz"
    "/path/to/output3.npz"
)

# Path to your Python script
PYTHON_SCRIPT="beat_this/beat_augmenter.py"

# Number of augmentations
NUM_AUG=4

# Noise level
NOISE=0.4

# Device
DEVICE=0

# Check if the number of dataset paths matches the number of output paths
if [ "${#data_paths[@]}" -ne "${#output_paths[@]}" ]; then
    echo "Error: Number of dataset paths and output paths must be the same!"
    exit 1
fi

# Iterate through the dataset paths and their corresponding output paths
for i in "${!data_paths[@]}"; do
    dataset="${data_paths[$i]}"
    output="${output_paths[$i]}"
    
    echo "Processing dataset: $dataset"
    echo "Saving output to: $output"
    
    # Run the Python script
    python "$PYTHON_SCRIPT" "$dataset" --path-to-npz "$output" --num-aug "$NUM_AUG" --noise "$NOISE" --gpu "$DEVICE"
    
    # Check if the script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error processing $dataset. Skipping to the next."
    fi

done

echo "All datasets processed."
