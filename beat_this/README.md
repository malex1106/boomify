# Beat This!

An adapted version of the beat tracker framework from the ISMIR 2024 paper "[Beat This! Accurate Beat Tracking Without DBN Postprocessing](https://arxiv.org/abs/2407.21658)" by Francesco Foscarin, Jan Schlüter and Gerhard Widmer, extended to support **Boomerang augmentations**. The official implementation can be found [here](https://github.com/CPJKU/beat_this).

This README provides installation and training instructions for the <code>beat_this</code> framework.

## Installation
Ensure you follow all the requirements outlined in the official beat tracker repository. Then, install our modified framework using:

```bash
% pip install -e .
```

To train a model, adhere to the specified folder structure so that all necessary data is accessible (annotations, spectrograms). In addition to the official implementation, make sure the .npz files containing the Boomerang augmentations are placed in the same directory as the other spectrogram files.

## Training Beat Tracker

To support extended training with Boomerang augmentation, new training parameters were introduced. Boomerang augmentation can be disabled by using the <code>--no-boom-augmentation</code> flag. When enabled, all relevant boomerang parameters must be explicitly passed to <code>train.py</code>, for example:

```bash
% python launch_scripts/train.py --boom-noise $NOISE --boom_num $NUMBER_OF_AUGMENTATIONS_PER_FILE
```

The bash script <code>run_exp.sh</code> performs multiple experiments using different augmentation configurations:

```bash
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
```
