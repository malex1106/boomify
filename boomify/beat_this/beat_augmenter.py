"""
Boomify samples and export them as log-mel spectrograms for Beat This!
"""

import warnings 
warnings.filterwarnings('ignore') 

import os
import argparse
import sys
import io
import torch
import torchaudio
import numpy as np
import soundfile as sf
from tqdm import tqdm
from zipfile import ZipFile

from boomify.inference import AudioProcessor
from boomify.dataset import AudioBatchProcessor
from boomify.utils import list_audiofiles


# Copied from https://github.com/CPJKU/beat_this/blob/main/beat_this/preprocessing.py
class LogMelSpect(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=441,
        f_min=30,
        f_max=11000,
        n_mels=128,
        mel_scale="slaney",
        normalized="frame_length",
        power=1,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        self.spect_class = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=normalized,
            power=power,
        ).to(device)
        self.log_multiplier = log_multiplier

    def forward(self, x):
        """Input is a waveform as a monodimensional array of shape T,
        output is a 2D log mel spectrogram of shape (F,128)."""
        return torch.log1p(self.log_multiplier * self.spect_class(x).T)
    

def get_parser():
    parser = argparse.ArgumentParser(
        description="Applies boomerang sampling to given audiofiles with the Stable Audio Open model."
    )
    parser.add_argument(
        "path",
        type=str,
        help="A path to an audio file to process."
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.5,
        help="Noise schedule for Boomerang Sampling."
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Percentage of overlap for latent blending."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Music",
        help="Text-prompt for Stable Audio Open."
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="Low quality",
        help="Negative text-prompt for Stable Audio Open."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference."
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=1,
        help="Number of augmentations per file."
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.,
        help="Guidance-scale for Stable Audio Open."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=100,
        help="Number of inference steps for Stable Audio Open."
    )
    parser.add_argument(
        "--path-to-npz",
        type=str,
        default=None,
        help="Path to .npz data file."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Which GPU to use, or -1 for CPU."
    )
    return parser


def run(
    path, noise, overlap, prompt, negative_prompt, batch_size, num_aug, guidance_scale, num_inference_steps, path_to_npz, gpu
):
    # Check if path_to_npz is given
    if path_to_npz is None:
        raise FileNotFoundError("Path to .npz not specified")
    
    # Get dataset name
    if path.endswith('/') or path.endswith('\\'):
        path = path[:-1]
    dataset_name = os.path.basename(path)

    # Determine device
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    audio_start = 0.0
    verbose = True

    # Prepare pipeline
    pipeline = AudioProcessor(
        noise, 
        overlap, 
        num_inferene_steps=num_inference_steps, 
        device=device, 
        verbose=verbose
    )

    model = pipeline.model
    target_sr_sao = model.vae.sampling_rate
    vae_length = int(model.transformer.config.sample_size) * model.vae.hop_length / target_sr_sao
    overlap_s = vae_length * overlap

    file_paths = list_audiofiles(path)

    print("---" * 20)
    print(f"Dataset name: {dataset_name}")
    print(f"Noise: {noise*100}%")
    print(f'Prompt: "{prompt}"')
    print(f'Negative-prompt: "{negative_prompt}"')
    print(f"Latent blending/ overlap: {overlap*100}%")
    print(f"Number of files: {len(file_paths)}\n")
    
    processor = AudioBatchProcessor(
        file_paths=file_paths,
        batch_size=batch_size,
        vae_length=vae_length,
        overlap=overlap,
        target_sr=target_sr_sao,
    )

    audio_sr_out = 22050

    resample = torchaudio.transforms.Resample(
        orig_freq=target_sr_sao,
        new_freq=audio_sr_out
    )

    mel_args = dict(
        n_fft=1024,
        hop_length=441,
        f_min=30,
        f_max=11000,
        n_mels=128,
        mel_scale="slaney",
        normalized="frame_length",
        power=1,
    )

    logspect_class = LogMelSpect(audio_sr_out, **mel_args)

    with ZipFile(path_to_npz, "a") as z:
        existing_folders = {name.split("/")[0] for name in z.namelist() if "/" in name}
        existing_files = {name for name in z.namelist() if "/" in name}

        # Iterate through batch (wave files)
        for batch, lengths, file_names in tqdm(processor, desc="Inference"):
            # Generate set of current file targets for the batch
            #batch_files = {f"{dataset_name}_{os.path.splitext(os.path.basename(name))[0]}" for name in file_names}
            batch_files = {f"{os.path.splitext(os.path.basename(name))[0]}" for name in file_names}
            
            # Check if any file from the batch exists
            if not batch_files.intersection(existing_folders):
                print("All files in batch are missing, skipping")
                continue
            
            # Iterate through number of augmentations per wave file
            for cur_aug in range(1, num_aug + 1):
                outputs = []
                previous_latent_tail = None
                # Iterate through window sequences of wave files
                for seq in range(batch.shape[1]):
                    audio_output, previous_latent_tail = pipeline(
                        batch[:, seq, ...], 
                        [prompt for _ in range(batch.shape[0])],
                        [negative_prompt for _ in range(batch.shape[0])],
                        previous_latent_tail,
                        guidance_scale=guidance_scale,
                        audio_start=audio_start,
                        audio_end=vae_length
                    )

                    audio_output = audio_output.audios.float().cpu()
                    outputs.append(audio_output)

                # Concat window sequences
                if overlap > 0:
                    for i in range(1, len(outputs)):
                        outputs[i] = outputs[i][:, :, int(overlap_s * target_sr_sao):]  # Trim the overlap part of the current output
                generated = torch.concatenate(outputs, axis=-1)
                
                # Convert to mono
                generated = torch.mean(generated, axis=1)

                # Save as .npy
                for i in range(len(lengths)):
                    file_name = os.path.basename(file_names[i])
                    #file = f"{dataset_name}_{os.path.splitext(file_name)[0]}"
                    file = f"{os.path.splitext(file_name)[0]}"

                    if file not in existing_folders:
                        print(f"{file} does not exist, skipping")
                        continue

                    path_to_augmentation = f"{file}/track_bs{int(noise*100)}_{cur_aug}.npy"

                    if path_to_augmentation in existing_files:
                        print(f"{path_to_augmentation} does already exist, skipping")
                        continue

                    audio = generated[i, :lengths[i]].to(torch.float32)
                    audio_resampled = resample(audio)
                    spectrogram = logspect_class(audio_resampled).cpu().numpy().astype(np.float16)

                    # Save the array to an in-memory buffer
                    with io.BytesIO() as buffer:
                        np.save(buffer, spectrogram)
                        buffer.seek(0)
                        z.writestr(path_to_augmentation, buffer.read())

                    # Optional audio file saving
                    #augmented_audio_path = os.path.join("augmentations", f"{file}_track_bs{int(noise*100)}_{cur_aug}.wav")
                    #sf.write(augmented_audio_path, audio_resampled, audio_sr_out)


def main():
    parser = get_parser()
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    sys.exit(main())