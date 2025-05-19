import glob
import os.path
from pathlib import Path


MY_EXTS = {'.flac', '.wav', '.mp3', '.ogg'}

def peak_normalize(audio):
    # Calculate the maximum absolute value
    max_val = audio.abs().max()

    # Avoid division by zero
    if max_val > 0:
        normalized_audio = audio / max_val
    else:
        normalized_audio = audio  # If the audio is silent (all zeros), return it unchanged

    return normalized_audio


def list_audiofiles(directory):
    # If the path is a file, check if it's an audio file.
    if os.path.isfile(directory):
        if any(directory.endswith(ext) for ext in MY_EXTS):
            return [directory]
        else:
            return []
    
    # Efficiently get both .wav and .flac files using a single pattern
    files = []
    for ext in MY_EXTS:
        files += glob.glob(f"{directory}/**/*{ext}", recursive=True)
    return files
