"""Component 8 (V2): better chroma features, as plain functions.

V2 changes the features only. chroma_cqt does CQT -> fold to 12 -> threshold -> normalize in one
call; V2 computes the CQT magnitude itself, changes it, and hands it back to chroma_cqt(C=...), so
with every V2 step switched off the result is the V0/V1 chroma (checked on val: max difference 3e-7).

    extract(path, cfg)                 audio file -> (treble, bass), two (12, n_frames) chromas
    cqt_magnitude(y, sr, tuning, cfg)  harmonic waveform -> (252, n_frames) |CQT|, C1..B7, 36 bins/octave
    bin_frequencies(n_bins, tuning)    centre frequency of every CQT bin, in Hz
    take_floor(C, cfg)                 per-bin percentile over the whole take
    remove_floor(C, freqs, cfg)        subtract that floor, only below cfg["fmax"] Hz (the body resonance)
    log_compress(C, cfg)               divide by the take level, then log(1 + gamma * C)
    fold_chroma(C, cfg)                (252, n_frames) -> (12, n_frames): chroma_cqt's fold, threshold, norm
    bass_chroma(C, freqs, cfg)         fold only the bins from E1 to B3, the low register of any band
"""
from pathlib import Path

import librosa
import numpy as np

from chordtracker.features import load_audio

BINS_PER_OCTAVE = 36
N_OCTAVES = 7
N_BINS = N_OCTAVES * BINS_PER_OCTAVE     # 252
FMIN = librosa.note_to_hz("C1")          # 32.7 Hz, chroma_cqt's default


# ====================================================================================
# ================================       CQT        ==================================
# ====================================================================================

def cqt_magnitude(y: np.ndarray, sr: int, tuning: float, cfg: dict) -> np.ndarray:
    """|CQT| exactly as chroma_cqt computes it: 7 octaves from C1, 36 bins per octave.

    tuning is in fractions of a bin (librosa.estimate_tuning with 36 bins per octave). Returns
    (252, n_frames); bin k is centred at bin_frequencies(252, tuning)[k].
    """
    return np.abs(librosa.cqt(y, sr=sr, hop_length=cfg["features"]["hop_length"], fmin=FMIN, n_bins=N_BINS,
                              bins_per_octave=BINS_PER_OCTAVE, tuning=tuning))


def bin_frequencies(n_bins: int = N_BINS, tuning: float = 0.0) -> np.ndarray:
    """Centre frequency in Hz of each CQT bin. Example: bin 0 = 32.7 Hz (C1), bin 3 = 34.6 Hz (C#1)."""
    return librosa.cqt_frequencies(n_bins, fmin=FMIN * 2.0 ** (tuning / BINS_PER_OCTAVE),
                                   bins_per_octave=BINS_PER_OCTAVE)


# ====================================================================================
# ================================  FLOOR REMOVAL   ==================================
# ====================================================================================

def take_floor(C: np.ndarray, cfg: dict) -> np.ndarray:
    """The per-bin floor of one take: the cfg["percentile"]-th percentile of every bin, shape (n_bins, 1).

    A bin's 50th percentile (median) is the level it stays above for half of the frames. A resonance
    present in every frame is its own median; a note is in the median only if it sounds in that bin
    for more than half of the take.
    Example: one bin [0, 0, 5, 5, 5, 1, 1] -> median 1, 20th percentile 0.
    """
    return np.percentile(C, cfg["percentile"], axis=1, keepdims=True)


def remove_floor(C: np.ndarray, freqs: np.ndarray, cfg: dict) -> np.ndarray:
    """max(C - alpha * floor, 0) on the bins centred below cfg["fmax"] Hz; the other bins are unchanged.

    cfg is the "floor" section: enabled, percentile (50 = median), alpha (1.0 = all of the floor),
    fmax (200 Hz: the ~106 Hz guitar body resonance and its neighbours; null = every bin).
    On val, the median below 200 Hz removes 3% of the energy of the transcribed notes and 27% of
    the rest; over every bin it removes 7% of the notes for less gain (proposals/v2/ablation_1_floor.md).
    """
    if not cfg["enabled"]:
        return C
    removed = np.maximum(C - cfg["alpha"] * take_floor(C, cfg), 0.0)
    if cfg["fmax"] is None:
        return removed
    return np.where((freqs < cfg["fmax"])[:, None], removed, C)


# ====================================================================================
# ================================   COMPRESSION    ==================================
# ====================================================================================

def log_compress(C: np.ndarray, cfg: dict) -> np.ndarray:
    """log(1 + gamma * C / level), level = the take's cfg["level_percentile"]-th percentile of C.

    cfg is the "compression" section. Dividing by the take's own level first makes gamma mean the
    same on a quiet and a loud recording. gamma 10 is mild: loud low notes no longer outweigh the
    quieter chord tones, but noise bins stay small (gamma 1000 flattens everything; 59.5% on val).
    """
    if not cfg["enabled"]:
        return C
    level = max(float(np.percentile(C, cfg["level_percentile"])), 1e-10)
    return np.log1p(cfg["gamma"] * C / level)


# ====================================================================================
# ================================      FOLDING     ==================================
# ====================================================================================

def fold_chroma(C: np.ndarray, cfg: dict) -> np.ndarray:
    """(252, n_frames) -> (12, n_frames) with chroma_cqt's own fold, threshold and normalization.

    V2 sets cfg["features"]["threshold"] to null: chroma_cqt's threshold is absolute and applied
    before normalization, so it depends on the recording level (on V1 it zeroes ~60% of the values).
    """
    features_cfg = cfg["features"]
    norm = float(features_cfg["norm"]) if features_cfg["norm"] is not None else None
    return librosa.feature.chroma_cqt(C=C, bins_per_octave=BINS_PER_OCTAVE, fmin=FMIN,
                                      threshold=features_cfg["threshold"], norm=norm)


def bass_chroma(C: np.ndarray, freqs: np.ndarray, cfg: dict) -> np.ndarray:
    """Chroma of the low register only: bins centred from cfg["fmin"] to cfg["fmax"], folded, max-normalized.

    cfg is the "bass" section, with fmin and fmax as note names: E1 (41.2 Hz, the low E of a bass
    guitar) to B3 (247 Hz). The range is the low register of any band, not the guitar's (whose
    low E is E2, 82 Hz). The edges are widened by half a semitone so E1 and B3 themselves are in.
    A frame with no energy in the range stays all zeros.
    """
    low = librosa.note_to_hz(cfg["fmin"]) * 2 ** (-0.5 / 12)
    high = librosa.note_to_hz(cfg["fmax"]) * 2 ** (0.5 / 12)
    keep = ((freqs >= low) & (freqs <= high))[:, None]
    folded = librosa.feature.chroma_cqt(C=np.where(keep, C, 0.0), bins_per_octave=BINS_PER_OCTAVE,
                                        fmin=FMIN, threshold=None, norm=None)
    return librosa.util.normalize(folded, norm=np.inf, axis=0, threshold=1e-10)


# ====================================================================================
# ================================     EXTRACT      ==================================
# ====================================================================================

def extract(path: Path | str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """One audio file -> (treble, bass) frame chromas, both (12, n_frames) on the V0 frame grid.

    load_audio (harmonic part) -> tuning -> |CQT| -> floor -> level + log -> fold (treble, full range)
    and fold E1..B3 (bass). The tuning is estimated as chroma_cqt does when cfg["features"]["tuning"]
    is null, so the bin frequencies used by the floor and the bass range are the CQT's own.
    """
    y, sr = load_audio(path, cfg)
    tuning = cfg["features"]["tuning"]
    if tuning is None:
        tuning = float(librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE))
    freqs = bin_frequencies(N_BINS, tuning)
    C = cqt_magnitude(y, sr, tuning, cfg)
    C = log_compress(remove_floor(C, freqs, cfg["floor"]), cfg["compression"])
    return fold_chroma(C, cfg), bass_chroma(C, freqs, cfg["bass"])
