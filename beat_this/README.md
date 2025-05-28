# Beat This!
An adapted version of the beat tracker framework from the ISMIR 2024 paper "[Beat This! Accurate Beat Tracking Without DBN Postprocessing](https://arxiv.org/abs/2407.21658)" by Francesco Foscarin, Jan Schlüter and Gerhard Widmer, extended to support **Boomerang augmentations**. The official implementation can be found [here](https://github.com/CPJKU/beat_this).

Ensure you follow all the requirements outlined in the official beat tracker repository. Then, install our modified framework using:

```bash
% pip install -e .
```

To train a model, adhere to the specified folder structure so that all necessary data is accessible (annotations, spectrograms). In addition to the official implementation, make sure the .npz files containing the Boomerang augmentations are placed in the same directory as the other spectrogram files.