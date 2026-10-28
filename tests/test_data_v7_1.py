"""Behaviour of V7.1's input: the window of a frame ends at it, the CRNN's output for a frame never reads a
later column, training still learns, and the column-by-column loop gives the whole-song posteriors."""
import numpy as np
import pytest
import torch

from chordtracker import data_v7_1 as d
from chordtracker import model_v6, train_v7_1
from chordtracker.data_v5 import CENTRE_ROW, Song
from chordtracker.streaming_v7_1 import CausalStreamingCRNN


def random_song(n_frames=200, seed=0):
    rng = np.random.default_rng(seed)
    return Song("toy", rng.random((168, n_frames), dtype=np.float32), rng.integers(0, 24, n_frames))


@pytest.mark.parametrize("start", [-20, 0, 37, 150])
@pytest.mark.parametrize("k", [-5, 0, 6])
def test_the_window_of_frame_t_is_columns_t_minus_14_to_t(start, k):
    song = random_song()
    x = d.excerpt(song, start, k, length=64)
    assert x.shape == (144, 64 + 14)
    top = CENTRE_ROW - 2 * k
    for t in range(64):
        frame = start + t
        for j, col in enumerate(range(frame - 14, frame + 1)):
            expected = song.cqt[top:top + 144, col] if 0 <= col < 200 else np.zeros(144)
            assert np.array_equal(x[:, t + j], expected)


def test_a_frame_score_never_reads_a_later_column():
    torch.manual_seed(0)
    model = model_v6.ChordCRNN(bidirectional=False).eval()
    song = random_song(n_frames=60)
    later = Song("toy", song.cqt.copy(), song.targets)
    later.cqt[:, 31:] = np.random.default_rng(5).random((168, 29))
    a, b = train_v7_1.song_posteriors(model, song), train_v7_1.song_posteriors(model, later)
    np.testing.assert_allclose(a[:, :31], b[:, :31], atol=1e-6)
    assert not np.allclose(a[:, 31:], b[:, 31:])


def test_column_by_column_gives_the_whole_song_posteriors():
    torch.manual_seed(1)
    model = model_v6.ChordCRNN(bidirectional=False)
    song = random_song(n_frames=40)
    whole = train_v7_1.song_posteriors(model, song)
    s = CausalStreamingCRNN(model)
    streamed = np.stack([s.step(song.cqt[CENTRE_ROW:CENTRE_ROW + 144, t]) for t in range(40)], axis=1)
    np.testing.assert_allclose(streamed, whole, atol=1e-5)


def one_note_song(roots, frames_per_note=10):
    """Toy song: each 'major chord' target is energy on its root's C3 row (as tests/test_train_v6.py)."""
    targets = np.repeat(roots, frames_per_note)
    cqt = np.zeros((168, len(targets)), dtype=np.float32)
    for t, r in enumerate(targets):
        if r >= 0:
            cqt[12 + 2 * (24 + r), t] = 1.0
    return Song("toy", cqt, targets)


def test_training_learns_toy_chords_and_keeps_the_best_epoch(monkeypatch):
    monkeypatch.setattr(train_v7_1, "device", lambda: torch.device("cpu"))
    cfg = {"train": {"seed": 0, "lr": 3e-3, "max_epochs": 15, "patience": 4, "dropout": 0.5,
                     "excerpt_frames": 16, "batch_excerpts": 4}}
    train_songs = [one_note_song([0, 4, 7, 9, -1] * 2, frames_per_note=8)] * 8
    val = {"toy": [one_note_song([2, 5, 11], frames_per_note=8)]}
    model, history = train_v7_1.train(train_songs, val, cfg)
    assert max(h["val_toy"] for h in history) >= 0.9
    assert not model.gru.bidirectional
    assert train_v7_1.frame_accuracy(model, val["toy"]) == max(h["val_toy"] for h in history)
