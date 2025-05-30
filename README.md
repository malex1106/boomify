# Music Boomerang: Reusing Diffusion Models for Data Augmentation and Audio Manipulation

Official implementation of the paper "<a href="">Music Boomerang: Reusing Diffusion Models for Data Augmentation and Audio Manipulation</a>", accepted to SMC 2025.

This work is conducted as part of Alexander Fichtinger's Master thesis at the [Institute of Computational Perception](https://www.jku.at/en/institute-of-computational-perception/) at JKU Linz, with Jan Schlüter as supervisor.

Check out [our project page](https://malex1106.github.io/boomify/) for demo audio samples.

## Online Demo

To try Boomerang sampling online, we provide a Google Colab notebook. [Open our example notebook in Google Colab](https://colab.research.google.com/github/malex1106/boomify/blob/main/boomify_example.ipynb) and follow the instructions.

## Overview
This repository contains two main components:
- [boomify](boomify/README.md) - The <code>boomify</code> package, which implements the **Music Boomerang** augmentation framework. It also includes tools for generating augmentations for the beat tracker framework, located at <code>boomify/beat_augmentation</code>. For installation and usage instructions, see the [boomify README](boomify/README.md).
- [beat_this](beat_this/README.md) - An adapted version of the original <code>beat_this</code> beat tracker framework, extended to support **Boomerang augmentations**. For installation and training instructions, follow the [beat_this README](beat_this/README.md).

To get started, download or clone the repository and follow the instructions provided in each components README file.

## Citation

```bibtex
@inproceedings{fichtinger2025musicboomerang,
    author = {Fichtinger, Alexander and Schl{\"u}ter, Jan and Widmer, Gerhard},
    title = {Music Boomerang: Reusing Diffusion Models for Data Augmentation and Audio Manipulation},
    year = 2025,
    month = jul,
    booktitle = {Proceedings of the 22nd Sound and Music Computing Conference},
}
```
