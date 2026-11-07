"""Component 19 (V7.1): the log-CQT computed from past audio only, as plain functions.

V5's log-CQT (features_v5) reads about 1.6 s of audio after each frame: the harmonic separation's
median and the CQT kernels are centred on their frame, and the level is the whole song's. V7.1
keeps every setting (grid, bins, gamma) and makes each step causal:

    causal_harmonic(y, sr, cfg)        harmonic part with the time median over the PAST 31 STFT frames
    bin_delays(cfg, tuning)            frames each CQT bin is delayed by: half its kernel, rounded up
    delay_bins(C, delays)              column t of bin k <- centred column t - d_k (zeros before the song)
    running_level(C, percentile)       the percentile of all values of frames 0..t, for every t
    start_tuning(y, cfg)               the tuning estimated on the first 10 s, and from which frame on
    causal_log_cqt(path, cfg)          audio file -> (168, n_frames) float32, causal
    cached_causal_log_cqt(...)         the same, computed once into data/processed/v7_1_logcqt/

What is left: the harmonic part's STFT window. A frame centred at sample n reads up to n + 1,024
and the overlap-add reaches one window further, so harmonic sample n depends on the audio up to
n + 2,048 = one CQT frame (LOOKAHEAD_FRAMES). Column t is therefore known once the audio of frame
t + 1 has arrived; reports/v7_1.md counts this frame in end-to-end WCSR.

CQT kernels: librosa sparsifies them by default (sparsity 0.01), which spreads each octave's kernels
over its whole FFT frame, about twice the kernel: C1 would then read 1.5 s ahead, not 0.53 s. V7.1
uses the exact kernels (sparsity 0), whose reach is at most half their length, so the bin delays hold.

Tuning: V5 estimates it on the whole file. scripts/v7_1_tuning.py found 23 % of the training songs more
than 10 cents from A440 (GuitarSet up to 25 cents), so V7.1 estimates it on the first
cfg["cqt"]["tuning_seconds"] (10 s) and uses A440 until that estimate exists (start_tuning). Since V10
the estimate reads +/-50 cents (tuning_in_bins) instead of wrapping at +/-25.
"""
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter

from chordtracker import paths

CACHE_DIR = paths.DATA_DIR / "processed" / "v7_1_logcqt"
N_FFT, STFT_HOP, MEDIAN_FRAMES = 2048, 512, 31     # librosa.effects.harmonic's defaults, as V0-V7
LOOKAHEAD_FRAMES = 1


def causal_harmonic(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """The harmonic part of y, as librosa.effects.harmonic but with the time median over past frames only.

    The time median covers cfg["audio"]["harmonic_median_frames"] past STFT frames (default 31, as V0-V7);
    a new note counts as harmonic once it fills half of them (31 frames: 0.37 s).

    Per STFT bin: H = median of the last 31 frames (frames before the song are silence), P = median
    over 31 neighbouring frequency bins of the same frame (no time involved, unchanged). The harmonic
    mask is softmask(H, margin * P), applied to the STFT, then inverted. Same length as y.
    """
    margin = cfg["audio"]["harmonic_margin"]
    S = librosa.stft(y, n_fft=N_FFT, hop_length=STFT_HOP)
    mag = np.abs(S)
    width = cfg["audio"].get("harmonic_median_frames", MEDIAN_FRAMES)
    H = median_filter(mag, size=(1, width), origin=(0, width // 2), mode="constant", cval=0.0)
    P = median_filter(mag, size=(MEDIAN_FRAMES, 1), mode="reflect")
    mask = librosa.util.softmask(H, margin * P, power=2.0, split_zeros=False)
    return librosa.istft(S * mask, hop_length=STFT_HOP, n_fft=N_FFT, length=len(y))


def bin_delays(cfg: dict, tuning: float = 0.0) -> np.ndarray:
    """(n_bins,) frames each bin is delayed by: ceil(half its kernel / hop). C1 6, A2 2, A3 and above 1."""
    q = cfg["cqt"]
    freqs = librosa.cqt_frequencies(q["n_bins"], fmin=librosa.note_to_hz(q["fmin"]),
                                    bins_per_octave=q["bins_per_octave"], tuning=tuning)
    lengths, _ = librosa.filters.wavelet_lengths(freqs=freqs, sr=cfg["audio"]["sr"], filter_scale=1)
    return np.ceil(lengths / 2 / q["hop_length"]).astype(int)


def delay_bins(C: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """Shift row k of C right by delays[k] columns, zeros in front: every kernel then ends at its frame.

    Example: row [1, 2, 3, 4], delay 2 -> [0, 0, 1, 2].
    """
    out = np.zeros_like(C)
    for k, d in enumerate(delays):
        if d < C.shape[1]:
            out[k, d:] = C[k, :C.shape[1] - d]
    return out


def running_level(C: np.ndarray, percentile: float) -> np.ndarray:
    """(n_frames,) level_t = the percentile of every value of columns 0..t (V2's level, from the past only).

    The last value equals np.percentile(C, percentile), V2's whole-song level.
    """
    n_bins, n = C.shape
    values = C.T.ravel()                                       # column 0's values first, then column 1's...
    out = np.array([np.percentile(values[:(t + 1) * n_bins], percentile) for t in range(n)])
    return np.maximum(out, 1e-10)


def causal_log_cqt(path: Path | str, cfg: dict) -> np.ndarray:
    """One audio file -> (n_bins, n_frames) float32 log-CQT whose column t uses audio up to frame t + 1."""
    y, _ = librosa.load(str(path), sr=cfg["audio"]["sr"], mono=cfg["audio"]["mono"])
    return causal_log_cqt_of_audio(y, cfg)


def start_tuning(y: np.ndarray, cfg: dict) -> tuple[float, int] | None:
    """(tuning, first frame it applies to), estimated on the first tuning_seconds of audio only; None if shorter.

    The estimate reads y[:n] (n = tuning_seconds of samples), as V5's estimate_tuning reads the whole
    file, at 12 bins per octave (tuning_in_bins; V10, before it 24 bins, which wrap at +/-25 cents). A column t is known once the audio up to frame t + 1 has arrived, so the estimate can be used
    from the first column whose frame t + 1 starts at or after sample n.
    """
    sr, q = cfg["audio"]["sr"], cfg["cqt"]
    n = int(q["tuning_seconds"] * sr)
    if len(y) <= n:
        return None
    return tuning_in_bins(causal_harmonic(y[:n], sr, cfg), sr, q["bins_per_octave"]), int(np.ceil(n / q["hop_length"])) - 1


def tuning_in_bins(y: np.ndarray, sr: int, bins_per_octave: int) -> float:
    """The tuning of y in fractions of a CQT bin at bins_per_octave, as librosa.cqt's `tuning` takes it.

    Estimated at 12 bins per octave, which reads -50 .. +50 cents, then converted (x bins_per_octave / 12).
    librosa.estimate_tuning at 24 bins reads only -25 .. +25 cents and wraps: a recording 35 cents flat
    came out as 15 cents sharp, its CQT a quarter tone off the semitone grid (issue #11).
    Example: 35 cents flat -> -0.35 at 12 bins -> -0.70 at 24 bins (50 cents a bin).
    """
    semitones = float(librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=12))
    return semitones * bins_per_octave / 12


def causal_log_cqt_of_audio(y: np.ndarray, cfg: dict) -> np.ndarray:
    """Harmonic part, CQT at A440 until the start estimate of the tuning is known and at that tuning from
    then on, bin delays, running level and log."""
    q, comp = cfg["cqt"], cfg["compression"]
    y_h = causal_harmonic(y, cfg["audio"]["sr"], cfg)
    C = delayed_cqt(y_h, cfg, 0.0)
    estimate = start_tuning(y, cfg)
    if estimate is not None:
        tuning, first = estimate
        C[:, first:] = delayed_cqt(y_h, cfg, tuning)[:, first:]
    level = running_level(C, comp["level_percentile"])
    return np.log1p(comp["gamma"] * C / level[None, :]).astype(np.float32)


def delayed_cqt(y_h: np.ndarray, cfg: dict, tuning: float) -> np.ndarray:
    """|CQT| of the harmonic signal at a tuning, exact kernels, every bin delayed so its kernel ends at its frame.

    The delays are taken at the larger of A440's and this tuning's kernels, so they hold for both.
    """
    q = cfg["cqt"]
    C = np.abs(librosa.cqt(y_h, sr=cfg["audio"]["sr"], hop_length=q["hop_length"],
                           fmin=librosa.note_to_hz(q["fmin"]), n_bins=q["n_bins"],
                           bins_per_octave=q["bins_per_octave"], tuning=tuning, sparsity=0))
    return delay_bins(C, np.maximum(bin_delays(cfg, 0.0), bin_delays(cfg, tuning)))


def cache_dir(cfg: dict) -> Path:
    """data/processed/v7_1_logcqt/, or v7_1_logcqt_median<n>/ for another harmonic median width."""
    width = cfg["audio"].get("harmonic_median_frames", MEDIAN_FRAMES)
    return CACHE_DIR if width == MEDIAN_FRAMES else CACHE_DIR.parent / f"v7_1_logcqt_median{width}"


def cached_causal_log_cqt(path: Path | str, key: str, cfg: dict) -> np.ndarray:
    """causal_log_cqt(path), saved under cache_dir(cfg) as <key>.npy (float16) the first time."""
    file = cache_dir(cfg) / f"{key}.npy"
    if file.exists():
        return np.load(file).astype(np.float32)
    C = causal_log_cqt(path, cfg)
    file.parent.mkdir(parents=True, exist_ok=True)
    np.save(file, C.astype(np.float16))
    return C
