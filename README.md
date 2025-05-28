# Music Boomerang: Reusing Diffusion Models for Data Augmentation and Audio Manipulation

Official implementation of the paper "<a href="">Music Boomerang: Reusing Diffusion Models for Data Augmentation and Audio Manipulation</a>", accepted to SMC 2025.

This work is conducted as part of Alexander Fichtinger's Master thesis at the [Institute of Computational Perception](https://www.jku.at/en/institute-of-computational-perception/) at JKU Linz, with Jan Schlüter as supervisor.

Find the demo samples [here]().

## Online Demo

To try Boomerang sampling online, we provide a Google Colab notebook. Open our example [notebook](https://colab.research.google.com/drive/1oVaC6aPEzQnOZ5t16jTWSa-1BfVHyeXr?usp=sharing) and follow the instructions!

## Overview
This repository contains two main components:
- <code>boomify</code> - The <code>boomify</code> package, which implements the **Music Boomerang** augmentation framework. It also includes tools for generating augmentations for the beat tracker framework, located at <code>boomify/beat_augmentation</code>.
- <code>beat_this</code> - An adapted version of the original <code>beat_this</code> beat tracker framework, extended to support **Boomerang augmentations**.

## Installation

For installation, you have to download or clone the repository. Then navigate to the <code>boomfiy</code> directory. Furthermore, a python-environment should be already installed (see <code>environment.yaml</code>). Then, activate the environment and use the following command:

```bash
% pip install -e .
```

### Connect with Huggingface
Stable Audio Open requires an account on Huggingface, and requesting access on [the model page](https://huggingface.co/stabilityai/stable-audio-open-1.0). After being granted access, create a token by opening the [access token settings](https://huggingface.co/settings/tokens), clicking ["Create new token"](https://huggingface.co/settings/tokens/new?tokenType=fineGrained), selecting "Fine-grained", and ticking the box for "Read access to contents of all public gated repos you can access". Copy the token and pass it directly from the command line using <code>huggingface-cli</code>:

```bash
% huggingface-cli login --token $HF_TOKEN
```

## Boomify Inference

To generate local samples of a set of audio files, you can use either the command line tool or Python's audio processor.

### Command Line
In addition to the python package, a command line application called <code>boomify</code> is available. For a full documentation, run:

```bash
% boomify --help
```

The basic usage is:

```bash
% boomify path/to/audios --noise 0.4 --output path/to/outputs --gpu 0
```

Using <code>--gpu=0</code> uses the first GPU, whereas <code>--gpu=-1</code> forces the use of the CPU.

### Python Class
If you are a python user, you can directly use the <code>boomify.inference</code> module.

Begin by instantiating the <code>AudioProcessor</code> class, which encapsulates the model and incorporates boomerang sampling:

```python
from boomify.inference import AudioProcessor
pipeline = AudioProcessor(noise, latent_overlap, num_inferene_steps, device, verbose)
```

To boomify an audio sample, run:
```python
audio_output, latent_tail = pipeline(audio_in, prompt, negative_prompt, previous_latent_tail, guidance_scale, audio_start, audio_end)
```

Please ensure your audio input has a sample rate matching the required standard (44.1 kHz), and consists of two channels. For frozen overlaps (latent blending), ensure that the previous latent tail is explicitly passed to the pipeline, with dimensions matching those specified during the pipeline instantiation, where <code>latent_overlap</code> is specified as a fraction (e.g., 0.25). The pipeline outputs the reconstructed audio waveform along with the current latent tail to be used in the next window iteration. See <code>boomify/boomify/cli.py</code> for an example implementation.

## Beat This!

You can use this framework to create variations/augmentations for the beat tracker from the ISMIR 2024 paper "[Beat This! Accurate Beat Tracking Without DBN Postprocessing](https://arxiv.org/abs/2407.21658)" by Francesco Foscarin, Jan Schlüter and Gerhard Widmer.

### Create Augmentations

We provide a Python script <code>beat_augmenter.py</code> that can be used to generate augmentations and save them as spectrograms in a specified .npz file. You can find this script in the <code>beat_augmentation</code> folder inside <code>boomify</code> (<code>boomify/beat_augmentation/beat_augmenter.py</code>).

It is essential that the filenames of the raw audio data match the  structure within the existing .npz files! Additionally, this script exclusively applies boomerang augmentations. All original audio spectrograms, as well as other augmented spectrograms (such as tempo, pitch and mask), must be downloaded or created separately!

Example usage:

```bash
% python beat_augmenter.py path/to/raw/audios --path-to-npz path/to/npz --noise 0.4 --num-aug 6 --gpu
```

The bash script <code>run_augmentations.sh</code> can be used to automatically iterate through multiple datasets. This script is also located at <code>boomify/beat_augmentation</code>.

If everything was successful, the generated .npz file for the guitarset should be structured accordingly, based on the specified number of augmentations per file and the provided noise level, e.g.:

```plaintext
guitarset_boom.npz/
│── 00_BN1-129-Eb_comp_mix/
│   │── track_bs40_1.npy
│   │── track_bs40_2.npy
│   │── track_bs40_3.npy
│   │── track_bs40_4.npy
│   │── track_bs40_5.npy
│   │── track_bs40_6.npy
|
...
```

### Training Beat Tracker

The original beat tracker framework has been extended with boomerang augmentation, available under the alias <code>beat_this</code>. For the following commands, navigate to the <code>beat_this</code> directory located at the root of the repository.

To support this, new training parameters were introduced. Boomerang augmentation can be disabled by using the <code>--no-boom-augmentation</code> flag. When enabled, all relevant boomerang parameters must be explicitly passed to <code>train.py</code>, for example:

```bash
% python launch_scripts/train.py --boom-noise $NOISE --boom_num $NUMBER_OF_AUGMENTATIONS_PER_FILE
```

To train a custom beat tracker model, follow the instructions at https://github.com/CPJKU/beat_this to set up the python-environment. Note that you need to install this adapted version, i.e., change directory to <code>beat_this</code> and run:

```bash
% pip install -e .
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
