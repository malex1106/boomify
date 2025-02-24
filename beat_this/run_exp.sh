#!/bin/bash

MAX_EPOCHS=600

for seed in 0 1 2; do
    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --no-tempo-augmentation --no-pitch-augmentation --no-mask-augmentation --no-boom-augmentation

    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --no-boom-augmentation

    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --no-tempo-augmentation --no-pitch-augmentation --no-mask-augmentation

    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val  
done
