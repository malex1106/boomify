"""
Implementation of AudioProcessor that handles model loading and boomerang sampling.
"""

import torch

from boomify.sampling import BommerangSampling
from boomify.model import load_model


class AudioProcessor(BommerangSampling):
    def __init__(
            self, 
            percent_noise,
            latent_overlap,
            num_inferene_steps=100,
            device="cpu",
            verbose=False, 
            generator=torch.Generator("cuda").manual_seed(0)
    ):
        self.device = torch.device(device)
        model = load_model(self.device, verbose=verbose)
        super().__init__(
            model, 
            percent_noise, 
            latent_overlap, 
            num_inferene_steps=num_inferene_steps, 
            generator=generator
        )