"""
Boomerang Sampling command line inference tool.
"""

import warnings 
warnings.filterwarnings('ignore') 

import time
import os
import argparse
import sys
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

from boomify.inference import AudioProcessor
from boomify.dataset import AudioBatchProcessor
from boomify.metrics import MatchingOnsets, MatchingBeatsAndTempo, MetricAverager
from boomify.utils import list_audiofiles

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
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for generated audio files. If omitted, outputs are saves next to input files."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Which GPU to use, or -1 for CPU."
    )
    return parser

def run(
    path, noise, overlap, prompt, negative_prompt, batch_size, guidance_scale, num_inference_steps, output, gpu
):
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
    target_sr = model.vae.sampling_rate
    vae_length = int(model.transformer.config.sample_size) * model.vae.hop_length / target_sr
    overlap_s = vae_length * overlap

    """
    file_paths = [
        r"D:\datasets\gtzan\test\blues.00000.wav",
        r"D:\datasets\gtzan\test\blues.00001.wav",
        r"D:\datasets\gtzan\test\\blues.00002.wav",
        r"D:\datasets\gtzan\test\blues.00003.wav"
    ]"""

    file_paths = list_audiofiles(path)

    print("---" * 20)
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
        target_sr=target_sr,
    )

    metric_manager = MetricAverager(
        MatchingOnsets(), 
        MatchingBeatsAndTempo()
    )

    for batch, lengths, file_names in tqdm(processor, desc="Inference"):
        previous_latent_tail = None
        outputs = []
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

            audio_output = audio_output.audios.float().cpu().numpy()
            outputs.append(audio_output)

        #start = time.time()
        batch_seq_list = [batch[:, seq, ...].cpu().numpy() for seq in range(batch.shape[1])]
        if overlap > 0:
            for i in range(1, len(outputs)):
                outputs[i] = outputs[i][:, :, int(overlap_s * target_sr):]  # Trim the overlap part of the current output
                batch_seq_list[i] = batch_seq_list[i][:, :, int(overlap_s * target_sr):]
        generated = np.concatenate(outputs, axis=-1)
        true_audio = np.concatenate(batch_seq_list, axis=-1)
        #print(time.time() - start)

        for i in range(len(lengths)):
            metric_manager.update(true_audio[i, :, :lengths[i]], generated[i, :, :lengths[i]])

            file_name = os.path.basename(file_names[i])
            file = os.path.splitext(file_name)
            
            if output is None:
                new_path = os.path.join(os.path.dirname(file_names[i]), f"{file[0]}.gen{file[1]}")
            else:
                new_path = os.path.join(output, f"{file[0]}.gen{file[1]}")

            sf.write(new_path, generated[i, :, :lengths[i]].T, target_sr)

    print(metric_manager.compute())


def main():
    parser = get_parser()
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    sys.exit(main())