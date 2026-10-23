"""Behaviour of the V5 log-CQT: grid, pitch position, compression, cache."""
import numpy as np
import soundfile as sf

from chordtracker import features_v5 as f5

CFG = {"audio": {"sr": 22050, "mono": True, "harmonic_margin": 4.0},
       "cqt": {"hop_length": 2048, "fmin": "F#0", "bins_per_octave": 24, "n_bins": 168},
       "compression": {"enabled": True, "level_percentile": 99, "gamma": 10}}


def write_tone(path, freq, seconds=3.0, sr=22050):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, 0.5 * np.sin(2 * np.pi * freq * t), sr)
    return path


def test_shape_and_frame_rate(tmp_path):
    C = f5.log_cqt(write_tone(tmp_path / "a.wav", 440.0), CFG)
    assert C.shape[0] == 168
    assert abs(C.shape[1] - 3.0 * 22050 / 2048) <= 2            # ~10.8 frames per second
    assert C.dtype == np.float32 and C.min() >= 0               # log1p of a magnitude is never negative


def test_a4_peaks_at_its_bin(tmp_path):
    """A4 (440 Hz) is 51 semitones above F#0, so its energy peaks at bin 2 * 51 = 102."""
    C = f5.log_cqt(write_tone(tmp_path / "a4.wav", 440.0), CFG)
    assert C[:, 10:-10].mean(axis=1).argmax() in (102, 103)
    assert f5.bin_notes(CFG)[51] == "A4"


def test_bin_notes_cover_f_sharp_0_to_f7():
    notes = f5.bin_notes(CFG)
    assert notes[0] == "F#0" and notes[-1] == "F7" and len(notes) == 84


def test_cache_returns_the_same_array(tmp_path, monkeypatch):
    monkeypatch.setattr(f5, "CACHE_DIR", tmp_path / "cache")
    wav = write_tone(tmp_path / "c.wav", 261.63)
    first = f5.cached_log_cqt(wav, "test/c4", CFG)
    assert (tmp_path / "cache" / "test" / "c4.npy").exists()
    again = f5.cached_log_cqt(wav, "test/c4", CFG)
    assert np.allclose(first, again, atol=1e-2)                 # float16 on disk
