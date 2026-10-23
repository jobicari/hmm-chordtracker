"""Component 13 (V5): the CNN's input, a log-CQT that is NOT folded into chroma, as plain functions.

V2 computes a CQT, compresses it and folds it into 12 pitch classes. V5 keeps the CQT itself, so the
network sees in which octave every note sounds (the bass note tells the root):

    log_cqt(path, cfg)              audio file -> (168, n_frames) float32 log-CQT, 2 bins per semitone
    bin_notes()                     note name of every second bin, for plots and checks
    cached_log_cqt(path, key, cfg)  log_cqt, computed once and kept in data/processed/v5_logcqt/

The grid, all from cfg["cqt"]:
    hop_length 2048 at 22,050 Hz = 93 ms per frame (10.8 frames/s): chords last ~1.6 s (RWC-P median
                segment), so a finer grid only repeats near-identical frames
    fmin F#0 (23.1 Hz), 24 bins per octave, 168 bins = 7 octaves, up to F7 (2,794 Hz)
                The network sees 144 of them (6 octaves, C1 to B6). The extra 6 semitones below and above
                are the room the pitch-shift crop slides in (data_v5.crop): shift k -> rows 12 - 2k .. +144.
Compression is V2's (features_v2.log_compress): divide by the take's 99th percentile, then log(1 + 10 C).
V2's floor and bass chroma are not used: they were built for chroma, and the CNN sees the low
register itself. The audio is the harmonic part (features.load_audio), as in every version since V0.
"""
from pathlib import Path

import librosa
import numpy as np

from chordtracker import paths
from chordtracker.features import load_audio
from chordtracker.features_v2 import log_compress

CACHE_DIR = paths.DATA_DIR / "processed" / "v5_logcqt"


def log_cqt(path: Path | str, cfg: dict) -> np.ndarray:
    """One audio file -> (n_bins, n_frames) float32 log-CQT.

    load_audio (harmonic part) -> tuning (estimated per file at the CQT's resolution) -> |CQT| ->
    level + log1p (V2). Example: a 241 s song -> (168, 2596).
    """
    q = cfg["cqt"]
    y, sr = load_audio(path, cfg)
    tuning = float(librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=q["bins_per_octave"]))
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=q["hop_length"], fmin=librosa.note_to_hz(q["fmin"]),
                           n_bins=q["n_bins"], bins_per_octave=q["bins_per_octave"], tuning=tuning))
    return log_compress(C, cfg["compression"]).astype(np.float32)


def bin_notes(cfg: dict) -> list[str]:
    """Note name of each semitone row (every second bin): ['F#0', 'G0', ..., 'F7']."""
    q = cfg["cqt"]
    n_semitones = q["n_bins"] * 12 // q["bins_per_octave"]
    start = librosa.note_to_midi(q["fmin"])
    return [librosa.midi_to_note(start + s, unicode=False) for s in range(n_semitones)]


def cached_log_cqt(path: Path | str, key: str, cfg: dict) -> np.ndarray:
    """log_cqt(path), saved as data/processed/v5_logcqt/<key>.npy (float16) the first time.

    key names the take, e.g. "rwc/RWC_P003" or "guitarset/00_BN1-129-Eb_comp". The cache does not know
    the settings: after changing cfg["cqt"] or cfg["compression"], delete the folder.
    """
    file = CACHE_DIR / f"{key}.npy"
    if file.exists():
        return np.load(file).astype(np.float32)
    C = log_cqt(path, cfg)
    file.parent.mkdir(parents=True, exist_ok=True)
    np.save(file, C.astype(np.float16))
    return C
