"""Tests for chordtracker.features (component 2).

Written ahead of the code: each test skips until the function it needs exists, so the suite
stays green while features.py is still a stub. Interface agreed in the thread:
    load_audio(path, cfg) -> (y, sr)    mono, resampled to cfg["audio"]["sr"], harmonic part only
    chroma(y, sr, cfg)    -> (12, T)    chroma_cqt with the settings in cfg["features"]
"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from chordtracker import features

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "v0.yaml").read_text())
SR = CFG["audio"]["sr"]
HOP = CFG["features"]["hop_length"]


def need(name):
    fn = getattr(features, name, None)
    if fn is None:
        pytest.skip(f"features.{name} not written yet")
    return fn


def c_major(seconds=2.0, sr=SR):
    """C4, E4 and G4 as sines: pitch classes 0, 4 and 7."""
    t = np.arange(int(seconds * sr)) / sr
    return (sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.00)) / 3).astype(np.float32)


def test_load_audio_gives_mono_at_target_rate(tmp_path):
    load_audio = need("load_audio")
    sf.write(tmp_path / "x.wav", np.stack([c_major(1.0, 44100)] * 2, axis=1), 44100)    # 1 s stereo, 44.1 kHz
    y, sr = load_audio(tmp_path / "x.wav", CFG)
    assert sr == SR and y.ndim == 1
    assert abs(len(y) - SR) <= 1


def test_chroma_shape_and_frame_count():
    chroma = need("chroma")
    y = c_major()
    # frames are centred (librosa pads), so there is one frame per hop plus the one at t = 0
    assert chroma(y, SR, CFG).shape == (12, 1 + len(y) // HOP)


def test_chroma_peaks_on_the_chord_notes():
    chroma = need("chroma")
    c = chroma(c_major(), SR, CFG)
    assert set(np.argsort(c.mean(axis=1))[-3:]) == {0, 4, 7}
