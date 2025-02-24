"""
Download Stable Audio Open 1.0 and modify it for Boomerang sampling.
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers import AudioPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging


logger = logging.get_logger(__name__)

def load_model(device, verbose=False):
    pipeline = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipeline = pipeline.to(device)

    if verbose:
        print("Params for vae: ", sum(p.numel() for p in pipeline.vae.parameters() if p.requires_grad))
        print("Params for text_encoder: ", sum(p.numel() for p in pipeline.text_encoder.parameters() if p.requires_grad))
        print("Params for transformer: ", sum(p.numel() for p in pipeline.transformer.parameters() if p.requires_grad))
        #print("Scheduler: ", type(pipeline.scheduler).__name__)
        print("Cuda available: ", torch.cuda.is_available())
        print("Autoencoder downsampling ratios: ", pipeline.vae.downsampling_ratios)

    type(pipeline).prepare_latents = prepare_latents
    type(pipeline).__call__ = __call__
    type(pipeline.scheduler).add_noise = add_noise_boomerang
    return pipeline


def add_noise_boomerang(self, original_samples, noise, timestep_index):
    # Make sure sigmas have the same device and dtype as original_samples
    sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    
    # add noise is called before first denoising step to create initial latent(img2img)
    step_indices = torch.tensor([timestep_index] * original_samples.shape[0])
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigma
    return noisy_samples


# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py#L411
# And modified for boomerang sampling by Alexander Fichtinger
def prepare_latents(
        self,
        batch_size,
        num_channels_vae,
        sample_size,
        dtype,
        device,
        generator,
        latents=None,
        initial_audio_waveforms=None,
        num_waveforms_per_prompt=None,
        audio_channels=None,
    ):
        shape = (batch_size, num_channels_vae, sample_size)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        # => not needed for boomerang since latents are already noisy! => only if latents are None!
        #latents = latents * self.scheduler.init_noise_sigma

        # encode the initial audio for use by the model
        if initial_audio_waveforms is not None:
            # check dimension
            if initial_audio_waveforms.ndim == 2:
                initial_audio_waveforms = initial_audio_waveforms.unsqueeze(1)
            elif initial_audio_waveforms.ndim != 3:
                raise ValueError(
                    f"`initial_audio_waveforms` must be of shape `(batch_size, num_channels, audio_length)` or `(batch_size, audio_length)` but has `{initial_audio_waveforms.ndim}` dimensions"
                )

            audio_vae_length = self.transformer.config.sample_size * self.vae.hop_length
            audio_shape = (batch_size // num_waveforms_per_prompt, audio_channels, audio_vae_length)

            # check num_channels
            if initial_audio_waveforms.shape[1] == 1 and audio_channels == 2:
                initial_audio_waveforms = initial_audio_waveforms.repeat(1, 2, 1)
            elif initial_audio_waveforms.shape[1] == 2 and audio_channels == 1:
                initial_audio_waveforms = initial_audio_waveforms.mean(1, keepdim=True)

            if initial_audio_waveforms.shape[:2] != audio_shape[:2]:
                raise ValueError(
                    f"`initial_audio_waveforms` must be of shape `(batch_size, num_channels, audio_length)` or `(batch_size, audio_length)` but is of shape `{initial_audio_waveforms.shape}`"
                )

            # crop or pad
            audio_length = initial_audio_waveforms.shape[-1]
            if audio_length < audio_vae_length:
                logger.warning(
                    f"The provided input waveform is shorter ({audio_length}) than the required audio length ({audio_vae_length}) of the model and will thus be padded."
                )
            elif audio_length > audio_vae_length:
                logger.warning(
                    f"The provided input waveform is longer ({audio_length}) than the required audio length ({audio_vae_length}) of the model and will thus be cropped."
                )

            audio = initial_audio_waveforms.new_zeros(audio_shape)
            audio[:, :, : min(audio_length, audio_vae_length)] = initial_audio_waveforms[:, :, :audio_vae_length]

            encoded_audio = self.vae.encode(audio).latent_dist.sample(generator)
            encoded_audio = encoded_audio.repeat((num_waveforms_per_prompt, 1, 1))
            latents = encoded_audio + latents
        return latents


# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py#L484
# And modified for boomerang sampling by Alexander Fichtinger
@torch.no_grad()
def __call__(
    self,
    noise_idx: int,
    latent_overlap_steps: int,
    prompt: Union[str, List[str]] = None,
    audio_end_in_s: Optional[float] = None,
    audio_start_in_s: Optional[float] = 0.0,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_waveforms_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    initial_audio_waveforms: Optional[torch.Tensor] = None,   # => not used since audio conditions are directly given as latents!
    initial_audio_sampling_rate: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    negative_attention_mask: Optional[torch.LongTensor] = None,
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    callback_steps: Optional[int] = 1,
    output_type: Optional[str] = "pt",
):
    # 0. Convert audio input length from seconds to latent length
    downsample_ratio = self.vae.hop_length

    max_audio_length_in_s = self.transformer.config.sample_size * downsample_ratio / self.vae.config.sampling_rate
    if audio_end_in_s is None:
        audio_end_in_s = max_audio_length_in_s

    if audio_end_in_s - audio_start_in_s > max_audio_length_in_s:
        raise ValueError(
            f"The total audio length requested ({audio_end_in_s-audio_start_in_s}s) is longer than the model maximum possible length ({max_audio_length_in_s}). Make sure that 'audio_end_in_s-audio_start_in_s<={max_audio_length_in_s}'."
        )

    waveform_start = int(audio_start_in_s * self.vae.config.sampling_rate)
    waveform_end = int(audio_end_in_s * self.vae.config.sampling_rate)
    waveform_length = int(self.transformer.config.sample_size)

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        audio_start_in_s,
        audio_end_in_s,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        attention_mask,
        negative_attention_mask,
        initial_audio_waveforms,
        initial_audio_sampling_rate,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self.encode_prompt(
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        attention_mask,
        negative_attention_mask,
    )

    # Encode duration
    seconds_start_hidden_states, seconds_end_hidden_states = self.encode_duration(
        audio_start_in_s,
        audio_end_in_s,
        device,
        do_classifier_free_guidance and (negative_prompt is not None or negative_prompt_embeds is not None),
        batch_size,
    )

    # Create text_audio_duration_embeds and audio_duration_embeds
    text_audio_duration_embeds = torch.cat(
        [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
    )

    audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2)

    # In case of classifier free guidance without negative prompt, we need to create unconditional embeddings and
    # to concatenate it to the embeddings
    if do_classifier_free_guidance and negative_prompt_embeds is None and negative_prompt is None:
        negative_text_audio_duration_embeds = torch.zeros_like(
            text_audio_duration_embeds, device=text_audio_duration_embeds.device
        )
        text_audio_duration_embeds = torch.cat(
            [negative_text_audio_duration_embeds, text_audio_duration_embeds], dim=0
        )
        audio_duration_embeds = torch.cat([audio_duration_embeds, audio_duration_embeds], dim=0)

    bs_embed, seq_len, hidden_size = text_audio_duration_embeds.shape
    # duplicate audio_duration_embeds and text_audio_duration_embeds for each generation per prompt, using mps friendly method
    text_audio_duration_embeds = text_audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
    text_audio_duration_embeds = text_audio_duration_embeds.view(
        bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
    )

    audio_duration_embeds = audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
    audio_duration_embeds = audio_duration_embeds.view(
        bs_embed * num_waveforms_per_prompt, -1, audio_duration_embeds.shape[-1]
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_vae = self.transformer.config.in_channels

    latents = self.prepare_latents(
        batch_size * num_waveforms_per_prompt,
        num_channels_vae,
        waveform_length,
        text_audio_duration_embeds.dtype,
        device,
        generator,
        latents,
        initial_audio_waveforms,
        num_waveforms_per_prompt,
        audio_channels=self.vae.config.audio_channels,
    )

    # 6. Prepare extra step kwargs
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare rotary positional embedding
    rotary_embedding = get_1d_rotary_pos_embed(
        self.rotary_embed_dim,
        latents.shape[2] + audio_duration_embeds.shape[1],
        use_real=True,
        repeat_interleave_real=False,
    )

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

    # Latent blending
    latent_overlap = latents[:, :, :latent_overlap_steps].clone() if latent_overlap_steps != 0 else None

    #with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        #######################################################################################
        # BOOMERANG CODE:
        # Skip any steps until given noise level
        #progress_bar.set_description(f"Processing. {t}")
        if i < noise_idx:
            continue
        #######################################################################################

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.transformer(
            latent_model_input,
            t.unsqueeze(0),
            encoder_hidden_states=text_audio_duration_embeds,
            global_hidden_states=audio_duration_embeds,
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # again add latent overlap
        if latent_overlap is not None:
            latents[:, :, :latent_overlap_steps] = latent_overlap

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            #progress_bar.update()
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

    # 9. Post-processing
    audio = self.vae.decode(latents).sample

    audio = audio[:, :, waveform_start:waveform_end]

    if output_type == "np":
        audio = audio.cpu().float().numpy()

    self.maybe_free_model_hooks()

    if not return_dict:
        return (audio,)

    if output_type == "latent":
        return AudioPipelineOutput(audios=audio), AudioPipelineOutput(audios=latents)
    else:
        return AudioPipelineOutput(audios=audio)