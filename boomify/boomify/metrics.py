import librosa
import mir_eval
import numpy as np
from collections import defaultdict 


def extract_onsets(audio, sr=44100, backtrack=False):
    """
    Extract onset times from an audio file (in seconds).
    """
    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, backtrack=backtrack)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


def eval_onsets(onsets1, onsets2, tolerance=0.08):
    """
    Computes the onset detection F-score.
    """
    return mir_eval.onset.f_measure(np.asarray(onsets1),
                                    np.asarray(onsets2),
                                    tolerance)[0]


def extract_beats(audio, sr=44100):
    """
    Extract tempo and beat times from an audio file.
    """
    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return tempo, beat_times


def compare_tempo(tempo1, tempo2):
    """
    Compare two tempos for equivalence.
    Relative tempo difference.
    """
    return abs(tempo1 - tempo2) / tempo1


def eval_beats(beats1, beats2, tolerance=0.08):
    """
    Computes the beat detection F-score.
    """
    return mir_eval.beat.f_measure(np.asarray(beats1),
                                   np.asarray(beats2),
                                   tolerance)


class Metric:
    """
    General interface for a metric.
    """

    def __call__(self, audio1, audio2):
        raise NotImplementedError("Subclass should implement this method.")
    

class MatchingOnsets(Metric):
    """
    Compute Structural Equivalence Score based on Onsets.
    """

    def __init__(self, tol=0.08, sr=44100, backtrack=False):
        self.tol = tol
        self.sr = sr
        self.backtrack = backtrack

    def __call__(self, audio1, audio2):
        assert audio1.shape == audio2.shape, "Audio shapes must match."
        scores = eval_onsets(
            extract_onsets(audio1, sr=self.sr, backtrack=self.backtrack), 
            extract_onsets(audio2, sr=self.sr, backtrack=self.backtrack), 
            tolerance=self.tol
        )
        return scores
    

class MatchingBeatsAndTempo(Metric):
    """
    Compares detected beats and tempo differences.
    """

    def __init__(self, tol=0.08, sr=44100):
        self.tol = tol
        self.sr = sr

    def __call__(self, audio1, audio2):
        audio1_tempo, audio1_beat_times = extract_beats(audio1)
        audio2_tempo, audio2_beat_times = extract_beats(audio2)

        tempo_score = compare_tempo(audio1_tempo, audio2_tempo)
        beat_score = eval_beats(audio1_beat_times, audio2_beat_times)
        return tempo_score, beat_score
    

class MetricAverager:
    """
    Class for averaging different metrics.
    """

    def __init__(self, *metrics):
        self.metrics = metrics
        self.state = defaultdict(list)

    def update(self, audio1, audio2):
        for metric in self.metrics:
            scores = metric(audio1, audio2)
            if isinstance(scores, tuple):
                for i, score in enumerate(scores):
                    metric_name = f"{metric.__class__.__name__}_{i+1}"
                    self.state[metric_name].append(score)
            else:
                metric_name = metric.__class__.__name__
                self.state[metric_name].append(scores)
    
    def compute(self):
        averages = {metric_name: (np.mean(val), np.var(val)) for metric_name, val in self.state.items()}
        return averages