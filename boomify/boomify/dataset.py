import torchaudio
import torch
import torch.nn.functional as F
import math


class AudioBatchProcessor:
    """
    Class for loading and pre-processing audio inputs. 
    """
    
    def __init__(self, file_paths, vae_length, batch_size=4, overlap=0.25, target_sr=44100):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.target_sr = target_sr
        self.vae_length = vae_length    # equals audio length (in seconds)
        self.overlap = vae_length * overlap
        self.files_info = self._precompute_file_info()

    def _precompute_file_info(self):
        files_info = []
        for file_path in self.file_paths:
            info = torchaudio.info(file_path)
            original_sr = info.sample_rate
            num_frames = info.num_frames

            # Calculate the total number of samples after resampling
            if original_sr != self.target_sr:
                num_samples = int(num_frames * (self.target_sr / original_sr))
            else:
                num_samples = num_frames

            files_info.append({
                "file_path": file_path,
                "original_sr": original_sr,
                "num_samples": num_samples,
                "waveform": None,  # Will be lazily loaded
            })
        return files_info
    
    def _load_file(self, file_info):
        file_path = file_info["file_path"]
        original_sr = file_info["original_sr"]

        # Load entire waveform
        waveform, _ = torchaudio.load(file_path)

        # Stereo
        if waveform.shape[0] < 2:  # Less than two channel
            waveform = waveform.repeat(2, 1)

        # Resample to target sample rate if necessary
        if original_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        file_info["waveform"] = waveform
        return waveform
    
    def __len__(self):
        return len(self.files_info)
    
    def __iter__(self):
        segment_length_samples = int(self.vae_length * self.target_sr)
        overlap_samples = int(self.overlap * self.target_sr)
        stride_samples = segment_length_samples - overlap_samples

        # Process in batches
        for batch_start in range(0, len(self), self.batch_size):
            batch_files = self.files_info[batch_start:batch_start + self.batch_size]

            # Load waveforms lazily
            for file_info in batch_files:
                if file_info["waveform"] is None:
                    self._load_file(file_info)

            # Determine the number of segments in the longest file of the batch
            max_segments = max(
                math.ceil(file_info["num_samples"] / stride_samples) for file_info in batch_files
            )

            total_lengths = torch.tensor([file_info["num_samples"] for file_info in batch_files])

            # Collect segments for each file in the batch
            batch_segments = []
            file_names = [file_info["file_path"] for file_info in batch_files]

            for file_info in batch_files:
                file_segments = []
                for segment_idx in range(max_segments):
                    start_idx = segment_idx * stride_samples
                    end_idx = start_idx + segment_length_samples

                    # Retrieve or pad the segment
                    if start_idx < file_info["num_samples"]:
                        segment = file_info["waveform"][:, start_idx:end_idx]
                        if segment.shape[1] < segment_length_samples:
                            pad_length = segment_length_samples - segment.shape[1]
                            segment = F.pad(segment, (0, pad_length))
                    else:
                        # All samples are processed; return a padded silence
                        segment = torch.zeros(2, segment_length_samples)

                    file_segments.append(segment)

                # Stack segments for the current file
                file_segments_tensor = torch.stack(file_segments)
                batch_segments.append(file_segments_tensor)

            # Combine batch segments into a single tensor
            batch_segments_tensor = torch.stack(batch_segments)
            yield batch_segments_tensor, total_lengths, file_names


if __name__ == "__main__":
    file_paths = [
        
    ]
    processor = AudioBatchProcessor(
        file_paths=file_paths,
        batch_size=2,
        vae_length=47.55446712018141,
        overlap=0.25,
        target_sr=44100,
    )

    for b1 in processor():
        print(b1[0].shape, b1[1])