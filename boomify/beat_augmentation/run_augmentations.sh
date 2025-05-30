#!/bin/bash

# Define the paths to your datasets and corresponding output files
data_paths=(
"/opt/datasets/fs/ballroom/audio"
"/opt/datasets/fs/gtzan/audio"
"/opt/datasets/fs/hainsworth/audio"
"/opt/datasets/fs/smc/audio"
"/opt/datasets/fs/beatles/audio"
"/opt/datasets/fs/simac/audio"
"/opt/datasets/fs/hjdb/audio"
"/opt/datasets/fs/rwc/audio"
"/opt/datasets/harmonixset/audio"
"/opt/datasets/tapcorrect/audio/flac"
"/opt/datasets/JAAH_2024/audio"
"/opt/datasets/filosax_beat/audio"
"/opt/datasets/asap-dataset/asap_beat/audio"
"/opt/datasets/groove_midi/beat_data/audio"
"/opt/datasets/guitarset/audio_mono-pickup_mix"
"/opt/datasets/candombe/candombe_audio"
)

output_paths=(
    "ballroom"
    "gtzan"
    "hainsworth"
    "smc"
    "beatles"
    "simac"
    "hjdb"
    "rwc"
    "harmonix"
    "tapcorrect"
    "jaah"
    "filosax"
    "asap"
    "groove_midi"
    "guitarset"
    "candombe"
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
    python "$PYTHON_SCRIPT" "$dataset" --path-to-npz "$output" --num-aug "$NUM_AUG" --noise "$NOISE" --gpu "$DEVICE" "${@:2}"
    
    # Check if the script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error processing $dataset. Skipping to the next."
    fi
done

echo "All datasets processed."
