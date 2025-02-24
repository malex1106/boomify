# Boomify (v0.1)

---

Implementation of Boomerang's local sampling strategy using Stable Audio Open 1.0.

This work is conducted as part of Alexander Fichtinger's Master thesis at the Institute of Computational Perception at JKU Linz, with Jan Schlüter as supervisor.

## Installation

For installation, you have to download or clone the repository. Furthermore, a python-environment should be already installed (see <code>environment.yaml</code>). Then, activate the environment and use the following command:

```bash
% pip install -e .
```

Note that you need to be authenticated to access the Stable Audio Open model weights. You can pass the token directly from the command line using <code>huggingface-cli</code>:

```bash
% huggingface-cli login --token $HF_TOKEN
```

## Boomify Inference

To generate local samples of a set of audio files, you can use either the command line tool or Python's audio processor.

In addition to the python package, a command line application called <code>boomify</code> is available. For a full documentation, run:

```bash
% boomify --help
```

The basic usage is:

```bash
% boomify path/to/audios --noise 0.4 --output path/to/outputs --gpu 0
```

Using <code>--gpu=1</code> uses the first GPU, whereas <code>--gpu=-1</code> forces the use of the CPU.

## Beat This!

You can use this framework to create variations/augmentations for the beat tracker from the ISMIR 2024 paper "[Beat This! Accurate Beat Tracking Without DBN Postprocessing](https://arxiv.org/abs/2407.21658)" by Francesco Foscarin, Jan Schlüter and Gerhard Widmer.

### Create Augmentations

We provide a Python script <code>beat_augmenter.py</code> that can be used to generate augmentations and save them as spectrograms in a specified .npz file.

It is essential that the filenames of the raw audio data match the existing structure within the .npz file! Additionally, this script exclusively applies boomerang augmentations. All original audio spectrograms, as well as other augmented spectrograms (such as tempo, pitch and mask), must be downloaded or created separately!

Example usage:

```bash
% python beat_this/beat_augmenter.py path/to/raw/audios --path-to-npz path/to/npz --noise $NOISE --num-aug $NUMBER_OF_AUGMENTATIONS_PER_FILE --gpu $DEVICE
```

The bash script <code>beat_this/run_augmentations.sh</code> can be used to automatically iterate through multiple datasets.

If everything was successful, you should see the following .npz structure for the guitarset, with two augmentations per file and a noise level of 0.4:

```plaintext
guitarset.npz/
│── 00_BN1-129-Eb_comp_mix/
│   │── track.npy
│   │── track_bs40_1.npy
│   │── track_bs40_2.npy
│   │── track_ps-1.npy
│   │── track_ps-2.npy
│   │── track_ps-3.npy
│   │── track_ps-4.npy
│   │── track_ps-5.npy
│   │── track_ps1.npy
│   │── track_ps2.npy
│   │── track_ps3.npy
│   │── track_ps4.npy
│   │── track_ps5.npy
│   │── track_ps6.npy
│   │── track_ts-4.npy
│   │── track_ts-8.npy
│   │── track_ts-12.npy
│   │── track_ts-16npy
│   │── track_ts-20.npy
│   │── track_ts4.npy
│   │── track_ts8.npy
│   │── track_ts12.npy
│   │── track_ts16npy
│   │── track_ts20.npy
|
...
```

### Training Beat Tracker

The original beat tracker framework has been extended with boomerang augmentation (saved as beat_this). To support this, an additional parameter was introduced for training. Boomerang augmentation can be disabled by using <code>--no-boom-augmentation</code>. The relevant boomerang parameters must be explicitly defined in <code>train.py</code>:

```python
if args.boom_augmentation:
    augmentations["boom"] = {"noise": 40, "num": 2}
```

To train a custom beat tracker model, follow the instructions at https://github.com/CPJKU/beat_this to set up the python-environment.

The bash script <code>run_exp.sh</code> performs multiple experiments using different augmentation configurations:

```bash
#!/bin/bash

MAX_EPOCHS=600

for seed in 0 1 2; do
    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --no-tempo-augmentation --no-pitch-augmentation --no-mask-augmentation --no-boom-augmentation
    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --no-boom-augmentation
    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val --no-tempo-augmentation --no-pitch-augmentation --no-mask-augmentation
    python launch_scripts/train.py --seed "$seed" --max-epochs "$MAX_EPOCHS" --no-val  
done
```
