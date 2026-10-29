"""Component 23 (V7.2): songs and excerpts with an input look-ahead, on the causal log-CQT.

V7.1's window of frame t is columns t - 14 .. t (no look-ahead). V7.2 lets it reach a frames past
its frame: columns t - 14 + a .. t + a, for an input look-ahead of a frames (a = 0 is V7.1). With
playback delayed by at least the total look-ahead (reports/v7_1.md, "The target set-up"), the label
still arrives with its frame. The CRNN (model_v6) is unchanged; only the strip is cut differently.

    load(take, cfg)                                -> Song on the causal log-CQT of cfg["cqt"]["fmin"]
    excerpt(song, start, k, length, lookahead)     (144, length + 14): columns start - 14 + a .. start + length - 1 + a
    make_batch(songs, starts, shifts, length, lookahead)

The CQT grid: from F#0 (V7.1's cache, reused) or from F#1, so that the network sees C2 to B7
(cache data/processed/v7_2_logcqt_<fmin>/). Targets, tiling and shifts are V6's.
"""
from pathlib import Path

import numpy as np
import torch

from chordtracker import data_v5, data_v6, data_v7_1, features_v7_1, paths
from chordtracker.acoustic_hmm import frame_times
from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS, PATCH_FRAMES, Song, Take

CONTEXT = PATCH_FRAMES - 1       # 15 columns per window


def cache_dir(cfg: dict) -> Path:
    """V7.1's cache for the F#0 grid; data/processed/v7_2_logcqt_<fmin>/ for another start."""
    fmin = cfg["cqt"]["fmin"]
    if fmin == "F#0":
        return features_v7_1.cache_dir(cfg)
    return paths.DATA_DIR / "processed" / f"v7_2_logcqt_{fmin.replace('#', 's')}"


def load(take: Take, cfg: dict) -> Song:
    """Cached causal log-CQT of the take on cfg's grid, plus the target at the centre time of every frame."""
    if cfg["cqt"]["fmin"] == "F#0":
        return data_v7_1.load(take, cfg)
    file = cache_dir(cfg) / f"{take.key}.npy"
    if file.exists():
        cqt = np.load(file).astype(np.float32)
    else:
        cqt = features_v7_1.causal_log_cqt(take.audio, cfg)
        file.parent.mkdir(parents=True, exist_ok=True)
        np.save(file, cqt.astype(np.float16))
    times = frame_times(cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    return Song(take.key, cqt, data_v5.frame_targets(take.annotation, times))


def excerpt(song: Song, start: int, k: int, length: int = data_v6.EXCERPT_FRAMES, lookahead: int = 0) -> np.ndarray:
    """(144, length + 14) input for frames start .. start + length - 1 with an input look-ahead of `lookahead`."""
    top = CENTRE_ROW - 2 * k
    first = start - CONTEXT + lookahead
    out = np.zeros((PATCH_BINS, length + CONTEXT), dtype=np.float32)
    lo, hi = max(first, 0), min(first + length + CONTEXT, song.cqt.shape[1])
    if lo < hi:
        out[:, lo - first:hi - first] = song.cqt[top:top + PATCH_BINS, lo:hi]
    return out


def make_batch(songs: list[Song], starts: np.ndarray, shifts: np.ndarray, length: int = data_v6.EXCERPT_FRAMES,
               lookahead: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs (B, 1, 144, length + 14) and targets (B, length), each excerpt shifted by its k."""
    x = np.stack([excerpt(songs[i], s, k, length, lookahead) for (i, s), k in zip(starts, shifts)])[:, None]
    y = np.stack([data_v6.excerpt_targets(songs[i], s, k, length) for (i, s), k in zip(starts, shifts)])
    return torch.from_numpy(x), torch.from_numpy(y).long()
