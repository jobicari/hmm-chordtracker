"""Behaviour of the V6 excerpts: same input as V5's patches, one shift per excerpt, every frame once per epoch."""
import numpy as np
import pytest
import torch

from chordtracker import data_v5
from chordtracker import data_v6 as d
from chordtracker.data_v5 import Song


def random_song(n_frames=300, seed=0):
    """Random CQT values and targets, with a stretch of -1 (no chord) in the middle."""
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, 24, n_frames)
    targets[100:140] = -1
    return Song("toy", rng.random((168, n_frames), dtype=np.float32), targets)


@pytest.mark.parametrize("start", [-50, 0, 37, 250])
@pytest.mark.parametrize("k", [-5, 0, 6])
def test_an_excerpt_is_v5_patches_side_by_side(start, k):
    song = random_song()
    x = d.excerpt(song, start, k, length=64)
    assert x.shape == (144, 64 + 14)
    for t in range(64):
        if 0 <= start + t < 300:                                 # data_v5.patch only takes frames of the song
            assert np.array_equal(x[:, t:t + 15], data_v5.patch(song, start + t, k))
        else:
            assert (x[:, t + d.MARGIN] == 0).all()               # a frame outside the song is zeros


def test_targets_are_shifted_and_minus_one_outside_the_song():
    song = random_song(n_frames=300)
    y = d.excerpt_targets(song, 250, 2, length=64)
    assert list(y[:50]) == list(data_v5.shift_targets(song.targets[250:], 2))
    assert (y[50:] == -1).all()
    assert (d.excerpt_targets(song, -20, 0, length=64)[:20] == -1).all()


def test_every_target_frame_is_in_exactly_one_excerpt_per_epoch():
    songs = [random_song(300, seed=1), random_song(90, seed=2), random_song(128, seed=3)]
    for seed in range(5):
        starts = d.excerpt_starts(songs, 64, np.random.default_rng(seed))
        seen = [np.zeros(len(s.targets), dtype=int) for s in songs]
        for i, s in starts:
            lo, hi = max(s, 0), min(s + 64, len(songs[i].targets))
            seen[i][lo:hi] += 1
        for song, count in zip(songs, seen):
            assert (count[song.targets >= 0] == 1).all()


def test_the_offset_moves_the_excerpt_boundaries_between_epochs():
    songs = [random_song(300)]
    offsets = {int(d.excerpt_starts(songs, 64, np.random.default_rng(s))[0, 1]) for s in range(20)}
    assert len(offsets) > 5 and all(-63 <= o <= 0 for o in offsets)


def test_excerpts_without_any_target_are_dropped():
    song = Song("toy", np.zeros((168, 200), dtype=np.float32), np.r_[np.full(130, -1), np.zeros(70, int)])
    starts = d.excerpt_starts([song], 64, np.random.default_rng(0))
    assert all(s + 64 > 130 for _, s in starts)


def one_note_song(roots, frames_per_note=10):
    """Toy song: each note is a 'major chord' target on its root, played as energy on the root's C3 row."""
    targets = np.repeat(roots, frames_per_note)
    cqt = np.zeros((168, len(targets)), dtype=np.float32)
    for t, r in enumerate(targets):
        if r >= 0:
            cqt[12 + 2 * (24 + r), t] = 1.0
    return Song("toy", cqt, targets)


def test_a_batch_moves_input_and_targets_together_with_one_shift_per_excerpt():
    songs = [one_note_song([0, 2, 9, 11, 4, 7], frames_per_note=12)]
    starts = np.array([[0, 0], [0, 20], [0, 40]])
    shifts = np.array([-5, 0, 6])
    x, y = d.make_batch(songs, starts, shifts, length=32)
    assert x.shape == (3, 1, 144, 32 + 14) and y.shape == (3, 32) and y.dtype == torch.long
    centre = x[:, 0, :, d.MARGIN:d.MARGIN + 32]                  # the column of each labelled frame
    keep = y >= 0
    note_row = centre.argmax(dim=1)
    assert torch.equal(((note_row // 2) % 12)[keep], (y % 12)[keep])


def test_the_whole_song_is_one_excerpt_at_inference():
    song = random_song(300)
    x = d.excerpt(song, 0, 0, length=300)
    assert x.shape == (144, 314)
    assert np.array_equal(x[:, d.MARGIN:-d.MARGIN], song.cqt[12:156])
    assert np.array_equal(d.excerpt_targets(song, 0, 0, length=300), song.targets)
