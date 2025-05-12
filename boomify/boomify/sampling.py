"""
Implementation for boomerang sampling.
"""

import torch
from diffusers.utils.torch_utils import randn_tensor
from boomify.utils import peak_normalize


class BoomerangSampling:
    """
    Class for performing boomerang sampling from Stable Audio Open.
    Note that the model pipeline also needs to be changed 
    e.g. prepare_latents, __call__, scheduler.add_noise.
    """
    
    def __init__(
            self, 
            model, 
            percent_noise,
            latent_overlap,
            num_inferene_steps, 
            generator
    ):
        self.generator = generator
        self.percent_noise = max(0.18, min(percent_noise, 1.0))

        self.model = model
        self.num_inference_steps = num_inferene_steps
        self.model.scheduler.set_timesteps(num_inferene_steps)
        self.audio_vae_length = int(model.transformer.config.sample_size) * model.vae.hop_length

        # Find the closest timestep corresponding to the desired percent_noise
        self.timestep_index = torch.abs(model.scheduler.timesteps - percent_noise).min(0)[1] # TODO: maybe set as property => then I could change noise level?

        # Latent blending
        overlap = self.audio_vae_length * latent_overlap
        timing_distance_per_latent = self.audio_vae_length / 1024
        self.latent_overlap_steps = int(overlap / timing_distance_per_latent)


    def _encode_latents(self, audio):
        with torch.no_grad():
            audio_tensor = audio.half().to(self.model.device)

            # Check shape
            audio_shape = (audio_tensor.shape[0] // 1, 2, self.audio_vae_length)    # (batch_size // num_waveforms_per_prompt(not used here), audio_channels, audio_vae_length)
            new_audio = audio_tensor.new_zeros(audio_shape)
            new_audio[:, :, :min(audio_tensor.shape[-1], self.audio_vae_length)] = audio_tensor[:, :, :self.audio_vae_length]

            # Peak normalize original audio
            new_audio = peak_normalize(new_audio)

            # Project audio into the latent space
            clean_z = self.model.vae.encode(new_audio).latent_dist.mode()
            return clean_z
        
    def _forward(self, latents):
        noise_tensor = randn_tensor(latents.shape, generator=self.generator, device=self.model.device, dtype=latents.dtype)

        noisy_latents = self.model.scheduler.add_noise(latents, noise_tensor, self.timestep_index)
        return noisy_latents

    def _reverse(
            self,
            prompt, 
            negative_prompt, 
            latents,
            latent_overlap_steps, 
            output_type,
            guidance_scale, 
            audio_start, 
            audio_end
    ):
        # Run the reverse boomerang process
        with torch.amp.autocast('cuda'):
            return self.model(
                self.timestep_index, 
                latent_overlap_steps,
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                latents=latents, 
                output_type=output_type,
                num_inference_steps=self.num_inference_steps, 
                guidance_scale=guidance_scale, 
                audio_start_in_s=audio_start, 
                audio_end_in_s=audio_end
            )
        
    def __call__(
            self, 
            audio_in, 
            prompt,
            negative_prompt,
            previous_latent_tail=None,  # for latent blending
            output_type="latent",
            guidance_scale=1, 
            audio_start=0., 
            audio_end=30.
    ):
        # 1. Encode latent
        clean_z = self._encode_latents(audio_in)
        
        # 2. Perform forward diffusion
        noisy_z = self._forward(clean_z)

        # Concentenate the previous tail (if not None) to the current noisy latent (latent blending)
        curr_latent_overlap = 0
        if previous_latent_tail is not None:
            noisy_z = torch.cat([previous_latent_tail, noisy_z[:, :, self.latent_overlap_steps:]], dim=-1)
            curr_latent_overlap = self.latent_overlap_steps

        # Ensure latent does not exceed T=1024
        assert noisy_z.shape[-1] == 1024, "time dimensions must be 1024"

        # 3. Perform reverse diffusion
        audio_output, latent = self._reverse(
            prompt, 
            negative_prompt, 
            noisy_z, 
            curr_latent_overlap,
            output_type,
            guidance_scale, 
            audio_start, 
            audio_end
        )

        del clean_z
        del noisy_z

        # Get new latent tail
        previous_latent_tail = latent.audios[:, :, -self.latent_overlap_steps:] if self.latent_overlap_steps != 0 else None
        return audio_output, previous_latent_tail