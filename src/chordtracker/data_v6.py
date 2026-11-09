"""Component 14 (V6): sequence input for the CRNN. Excerpts of consecutive frames instead of single patches.

V5 trained on one frame at a time: a 144 x 15 patch and the chord at its centre. A recurrent layer
learns from the ORDER of the frames, so V6 trains on excerpts: `length` consecutive frames of one
song, all shifted by the same k, with one target per frame.

    excerpt(song, start, k, length)           (144, length + 14) input: V5's patches side by side
    excerpt_targets(song, start, k, length)   (length,) targets of the same frames, shifted by k, -1 = none
    excerpt_starts(songs, length, rng)        one epoch's excerpts: (song index, start) tiling every song
    make_batch(songs, starts, shifts, length) (B, 1, 144, length + 14) inputs + (B, length) targets

The input has 7 extra frames on each side (14 in all) so that every one of the `length` frames keeps
the 15-frame context it had in V5: columns t .. t + 14 of an excerpt are exactly data_v5.patch(song,
start + t, k). The convolutional front end therefore sees the same numbers as V5's, and only what
comes after it (the recurrent layer) is new.

One shift per excerpt, not one per frame: a progression keeps its intervals when the whole excerpt is
moved k semitones (C -> G -> Am becomes D -> A -> Bm), so the order of chords the network sees is
still one that music uses. A new shift per frame would turn every chord change into a random jump.

Tiling: each song is cut into back-to-back excerpts from a random offset in -(length - 1) .. 0, a new
offset every epoch. Every frame with a target is in exactly one excerpt per epoch, as in V5's one pass
over the frames, and the excerpt boundaries fall in different places each epoch, so no chord change is
always cut off from the frames before it. Frames before the song start or past its end are zero input
and -1 targets (left out of the loss). Excerpts with no target at all are dropped.

At inference the whole song is one excerpt: excerpt(song, 0, 0, n_frames) gives (144, n_frames + 14).
"""
import numpy as np
import torch

from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS, PATCH_FRAMES, Song, shift_targets

EXCERPT_FRAMES = 128             # 11.9 s at 10.8 frames/s: several chord changes in every excerpt
MARGIN = PATCH_FRAMES // 2       # 7 frames of context before the first and after the last labelled frame


def excerpt(song: Song, start: int, k: int, length: int = EXCERPT_FRAMES) -> np.ndarray:
    """(144, length + 14) input for frames start .. start + length - 1, shifted up k semitones.

    Same crop as data_v5.patch (rows 12 - 2k .. 12 - 2k + 144); columns outside the song are zeros.
    """
    top = CENTRE_ROW - 2 * k
    first = start - MARGIN
    out = np.zeros((PATCH_BINS, length + 2 * MARGIN), dtype=np.float32)
    lo, hi = max(first, 0), min(first + length + 2 * MARGIN, song.cqt.shape[1])
    if lo < hi:
        out[:, lo - first:hi - first] = song.cqt[top:top + PATCH_BINS, lo:hi]
    return out


def excerpt_targets(song: Song, start: int, k: int, length: int = EXCERPT_FRAMES,
                    n_chords: int | None = None) -> np.ndarray:
    """(length,) targets of frames start .. start + length - 1, shifted by k; -1 outside the song.

    n_chords: classes from n_chords on (N) are not shifted (data_v5.shift_targets).
    """
    y = np.full(length, -1, dtype=int)
    lo, hi = max(start, 0), min(start + length, len(song.targets))
    if lo < hi:
        y[lo - start:hi - start] = song.targets[lo:hi]
    return shift_targets(y, k, n_chords)


def excerpt_starts(songs: list[Song], length: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 2) array of (song index, start frame): every song tiled from a random offset, empty tiles dropped.

    Example: a 300-frame song, length 128, offset -50 -> starts -50, 78, 206 (frames 0..299 covered once).
    """
    rows = []
    for i, song in enumerate(songs):
        n = len(song.targets)
        has_target = np.concatenate([[0], np.cumsum(song.targets >= 0)])     # targets in frames 0 .. t - 1
        for start in range(-int(rng.integers(0, length)), n, length):
            lo, hi = max(start, 0), min(start + length, n)
            if has_target[hi] > has_target[lo]:
                rows.append((i, start))
    return np.array(rows, dtype=int).reshape(-1, 2)


def make_batch(songs: list[Song], starts: np.ndarray, shifts: np.ndarray,
               length: int = EXCERPT_FRAMES) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs (B, 1, 144, length + 14) and targets (B, length) of the excerpts, each shifted by its k."""
    x = np.stack([excerpt(songs[i], s, k, length) for (i, s), k in zip(starts, shifts)])[:, None]
    y = np.stack([excerpt_targets(songs[i], s, k, length) for (i, s), k in zip(starts, shifts)])
    return torch.from_numpy(x), torch.from_numpy(y).long()
