"""Component 20 (V7.1): songs and excerpts on the causal log-CQT, with a window that ends at its frame.

V6's window of frame t is columns t - 7 .. t + 7 of the log-CQT. V7.1's is t - 14 .. t, so a frame
is labelled from what has been heard up to it. The CRNN (model_v6) is unchanged: it takes the 15
columns t .. t + 14 of an excerpt strip as the window of excerpt frame t, so V7.1 only changes how
the strip is cut: 14 frames of context before the first labelled frame, none after the last.

    load(take, cfg)                           -> Song: causal log-CQT (features_v7_1 cache) + targets
    excerpt(song, start, k, length)           (144, length + 14): columns start - 14 .. start + length - 1
    make_batch(songs, starts, shifts, length) as data_v6.make_batch, with this excerpt

Targets, tiling and shifts are V6's (data_v6.excerpt_targets, excerpt_starts). Columns before the
song are zeros, as in V6 and as a real-time system starts.
"""
import numpy as np
import torch

from chordtracker import data_v5, data_v6
from chordtracker.acoustic_hmm import frame_times
from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS, PATCH_FRAMES, Song, Take
from chordtracker.features_v7_1 import cached_causal_log_cqt

CONTEXT = PATCH_FRAMES - 1       # 14 frames before a frame complete its window


def load(take: Take, cfg: dict) -> Song:
    """Cached causal log-CQT of the take, plus the target at the centre time of every frame."""
    cqt = cached_causal_log_cqt(take.audio, take.key, cfg)
    times = frame_times(cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    return Song(take.key, cqt, data_v5.frame_targets(take.annotation, times))


def excerpt(song: Song, start: int, k: int, length: int = data_v6.EXCERPT_FRAMES) -> np.ndarray:
    """(144, length + 14) input for frames start .. start + length - 1, shifted up k semitones.

    Same crop as data_v5.patch (rows 12 - 2k .. 12 - 2k + 144); columns outside the song are zeros.
    """
    top = CENTRE_ROW - 2 * k
    first = start - CONTEXT
    out = np.zeros((PATCH_BINS, length + CONTEXT), dtype=np.float32)
    lo, hi = max(first, 0), min(start + length, song.cqt.shape[1])
    if lo < hi:
        out[:, lo - first:hi - first] = song.cqt[top:top + PATCH_BINS, lo:hi]
    return out


def make_batch(songs: list[Song], starts: np.ndarray, shifts: np.ndarray,
               length: int = data_v6.EXCERPT_FRAMES) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs (B, 1, 144, length + 14) and targets (B, length) of the excerpts, each shifted by its k."""
    x = np.stack([excerpt(songs[i], s, k, length) for (i, s), k in zip(starts, shifts)])[:, None]
    y = np.stack([data_v6.excerpt_targets(songs[i], s, k, length) for (i, s), k in zip(starts, shifts)])
    return torch.from_numpy(x), torch.from_numpy(y).long()
