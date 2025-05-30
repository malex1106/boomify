#!/bin/bash

# Define the paths to your datasets and corresponding output files
data_paths=(
"/opt/datasets/fs/ballroom/audio"
"/opt/datasets/candombe/candombe_audio"
"/opt/datasets/guitarset/audio_mono-pickup_mix"
"/opt/datasets/tapcorrect/audio/flac"
)

output_paths=(
    "ballroom"
    "candombe"
    "guitarset"
    "tapcorrect"
)

output_base="/opt/scratch/beat_this/spectrograms"

# Path to your Python script
PYTHON_SCRIPT="${0%/*}/beat_augmenter.py"

# Number of augmentations
NUM_AUG=8

# Noise level
NOISE="${2:-0.4}"

# Device
DEVICE="${1:-0}"

# Check if the number of dataset paths matches the number of output paths
if [ "${#data_paths[@]}" -ne "${#output_paths[@]}" ]; then
    echo "Error: Number of dataset paths and output paths must be the same!"
    exit 1
fi

# Iterate through the dataset paths and their corresponding output paths
for i in "${!data_paths[@]}"; do
    dataset="${data_paths[$i]}"
    name="${output_paths[$i]}"
    output="${output_base}/${name}_boom.npz"
    [ -f "${output%npz}lock" ] && continue
    touch "${output%npz}lock";
    echo "Processing dataset $dataset on GPU $DEVICE to $output"
    
    # Run the Python script
    python "$PYTHON_SCRIPT" "$dataset" --path-to-npz "$output" --num-aug "$NUM_AUG" --noise "$NOISE" --gpu "$DEVICE"
    
    # Check if the script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error processing $dataset. Skipping to the next."
    fi
done

echo "All datasets processed."
