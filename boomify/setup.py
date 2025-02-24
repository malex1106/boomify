from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    long_description = f.read()
setup(
    name="boomify",
    version="0.1",
    description="Boomify: Boomerang Local Sampling on Music Manifolds using Diffusion Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alexander Fichtinger",
    entry_points={
        "console_scripts": ["boomify=boomify.cli:main"],
    },
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=2",
        "torchaudio",
        "diffusers",
        "torchsde",
        "huggingface_hub",
        "accelerate",
        "soundfile",
        "librosa",
        "mir-eval"
    ],
    python_requires=">=3",
)