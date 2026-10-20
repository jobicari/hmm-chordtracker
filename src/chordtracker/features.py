"""Component 2: audio loading and frame-level chroma features.

Two plain functions, both driven by configs/v0.yaml:
    load_audio(path, cfg) -> (y, sr)    mono waveform at cfg["audio"]["sr"], harmonic part only
    chroma(y, sr, cfg)    -> (12, T)    chroma_cqt with the settings in cfg["features"]

Loading audio lives here, not in data: the manifest only points at audio files.
FeatureExtractor is the interface the rest of the pipeline uses; ChromaCQTExtractor wraps the two
functions with settings read once from the config, so every file is processed the same way.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import librosa
import numpy as np


# ====================================================================================
# ================================      AUDIO       ==================================
# ====================================================================================

def load_audio(path: Path | str, cfg: dict) -> tuple[np.ndarray, int]:
    """Read an audio file and return (waveform, sample_rate), ready for feature extraction.

    Settings come from cfg["audio"]:
        sr               every file is resampled to this rate (e.g. 44100 Hz on disk -> 22050 Hz)
        mono             True downmixes stereo: (n_samples, 2) on disk -> (n_samples,) in memory
        harmonic_margin  margin for librosa.effects.harmonic, which removes the percussive part
                         (strums, pick attacks) and keeps the sustained notes. null in the YAML
                         (None in Python) skips this step.

    The harmonic part has the same length as the input, so it never changes the frame count.
    """
    audio_cfg = cfg["audio"]
    target_sr = audio_cfg["sr"]
    harmonic_margin = audio_cfg["harmonic_margin"]

    waveform, sample_rate = librosa.load(str(path), sr=target_sr, mono=audio_cfg["mono"])
    if harmonic_margin is not None:
        waveform = librosa.effects.harmonic(waveform, margin=harmonic_margin)
    return waveform, sample_rate


# ====================================================================================
# ================================     CHROMA       ==================================
# ====================================================================================

def chroma(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Return the (12, n_frames) chroma of a waveform: how much energy each pitch class has per frame.

    Row i is pitch class i (0 = C, 1 = C#, ..., 11 = B), the same order as the chord templates.
    Column t is centred at t * hop_length / sr seconds (librosa pads the signal), so a waveform of
    n samples gives 1 + n // hop_length frames: 44100 samples with hop 512 -> 87 frames.

    Settings come from cfg["features"]:
        kind        only "chroma_cqt" exists in V0; anything else raises instead of running silently
        hop_length  samples between frames (512 at 22050 Hz = 23.2 ms)
        threshold   chroma_cqt sets values below this to 0 after normalization (removes weak noise)
        tuning      deviation from A440 in fractions of a bin; 0.0 = fixed tuning, no estimation
        norm        per-frame normalization; "inf" divides each column by its max, so the loudest
                    pitch class of every frame is 1.0
    """
    features_cfg = cfg["features"]
    if features_cfg["kind"] != "chroma_cqt":
        raise ValueError(f"unsupported feature kind {features_cfg['kind']!r}, expected 'chroma_cqt'")

    # YAML reads `inf` as the string "inf"; float() turns it into np.inf (and leaves 1 or 2 as numbers)
    norm = float(features_cfg["norm"]) if features_cfg["norm"] is not None else None

    return librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=features_cfg["hop_length"],
        threshold=features_cfg["threshold"],
        tuning=features_cfg["tuning"],
        norm=norm,
    )


# ====================================================================================
# ================================    EXTRACTORS    ==================================
# ====================================================================================

class FeatureExtractor(ABC):
    """Turns one audio file into a (n_features, n_frames) matrix. Subclasses must implement extract().

    Every extractor also knows its frame grid (sr and hop_length), so frame t can be matched to the
    chord sounding at frame_times()[t], e.g. with ChordAnnotation.at().
    """
    sr: int
    hop_length: int

    @abstractmethod
    def extract(self, path: Path | str) -> np.ndarray:
        """ Load the audio at path and return its features, one column per frame """

    def frame_times(self, n_frames: int) -> np.ndarray:
        """Centre time in seconds of each frame: frame t is at t * hop_length / sr (0.0, 0.0232, 0.0464, ...)."""
        return librosa.frames_to_time(np.arange(n_frames), sr=self.sr, hop_length=self.hop_length)




class ChromaCQTExtractor(FeatureExtractor):
    """V0 extractor: load_audio() then chroma() with fixed settings, giving a (12, n_frames) chroma."""

    def __init__(self, sr=22050, mono=True, harmonic_margin=4.0,
                 hop_length=512, threshold=0.25, tuning=0.0, norm=np.inf):
        self.sr = sr
        self.mono = mono
        self.harmonic_margin = harmonic_margin      # None skips HPSS
        self.hop_length = hop_length
        self.threshold = threshold
        self.tuning = tuning                        # None lets librosa estimate it per file
        self.norm = norm

    @classmethod
    def from_config(cls, cfg: dict):
        """Build the extractor from a loaded YAML config (cfg["audio"] and cfg["features"])."""
        audio_cfg, features_cfg = cfg["audio"], cfg["features"]
        # fail early on a config meant for another extractor, not after loading 180 files
        if features_cfg["kind"] != "chroma_cqt":
            raise ValueError(f"ChromaCQTExtractor needs features.kind 'chroma_cqt', got {features_cfg['kind']!r}")
        return cls(sr=audio_cfg["sr"], mono=audio_cfg["mono"], harmonic_margin=audio_cfg["harmonic_margin"],
                   hop_length=features_cfg["hop_length"], threshold=features_cfg["threshold"],
                   tuning=features_cfg["tuning"], norm=features_cfg["norm"])

    def config(self) -> dict:
        """The settings as a config dict, in the shape load_audio() and chroma() expect."""
        return {"audio": {"sr": self.sr, "mono": self.mono, "harmonic_margin": self.harmonic_margin},
                "features": {"kind": "chroma_cqt", "hop_length": self.hop_length, "threshold": self.threshold,
                             "tuning": self.tuning, "norm": self.norm}}

    def extract(self, path):
        cfg = self.config()
        waveform, sample_rate = load_audio(path, cfg)
        return chroma(waveform, sample_rate, cfg)
