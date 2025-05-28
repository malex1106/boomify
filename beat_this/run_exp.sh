#!/bin/bash

MAX_EPOCHS=400
BOOM_NOISE=40
BOOM_NUM=6

for seed in 0 1 2; do
    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --no-tempo-augmentation --no-pitch-augmentation --no-mask-augmentation --no-boom-augmentation --logger "wandb" --name "No-augmentation-$seed"

    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --no-boom-augmentation --logger "wandb" --name "Pitch-tempo-mask-$seed"

    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --no-tempo-augmentation --no-pitch-augmentation --no-mask-augmentation --boom-noise "$BOOM_NOISE" --boom-num "$BOOM_NUM" --logger "wandb" --name "Boomerang-sampling-$seed"

    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --boom-noise "$BOOM_NOISE" --boom-num "$BOOM_NUM" --logger "wandb" --name "All-augmentations-$seed"
done
