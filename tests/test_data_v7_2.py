"""Behaviour of V7.2's input look-ahead: the window of frame t is t - 14 + a .. t + a, the CRNN's output for
frame t never reads past column t + a, a column-by-column loop gives the whole-song posteriors, and training
still learns."""
import numpy as np
import pytest
import torch

from chordtracker import data_v7_1, data_v7_2, model_v6, train_v7_1, train_v7_2
from chordtracker.data_v5 import CENTRE_ROW, Song
from chordtracker.streaming_v7_2 import LookaheadStreamingCRNN


def random_song(n_frames=200, seed=0):
    rng = np.random.default_rng(seed)
    return Song("toy", rng.random((168, n_frames), dtype=np.float32), rng.integers(0, 24, n_frames))


@pytest.mark.parametrize("a", [0, 1, 4])
@pytest.mark.parametrize("start", [-20, 37, 150])
def test_the_window_of_frame_t_ends_a_frames_after_it(a, start):
    song, k = random_song(), 2
    x = data_v7_2.excerpt(song, start, k, length=64, lookahead=a)
    top = CENTRE_ROW - 2 * k
    for t in range(64):
        for j, col in enumerate(range(start + t - 14 + a, start + t + a + 1)):
            expected = song.cqt[top:top + 144, col] if 0 <= col < 200 else np.zeros(144)
            assert np.array_equal(x[:, t + j], expected)


def test_lookahead_zero_is_v7_1():
    song = random_song()
    assert np.array_equal(data_v7_2.excerpt(song, 30, -3, 64, 0), data_v7_1.excerpt(song, 30, -3, 64))
    torch.manual_seed(0)
    model = model_v6.ChordCRNN(bidirectional=False)
    np.testing.assert_allclose(train_v7_2.song_posteriors(model, song, 0), train_v7_1.song_posteriors(model, song))


@pytest.mark.parametrize("a", [1, 4])
def test_a_frame_score_never_reads_past_its_lookahead(a):
    torch.manual_seed(0)
    model = model_v6.ChordCRNN(bidirectional=False).eval()
    song = random_song(n_frames=60)
    later = Song("toy", song.cqt.copy(), song.targets)
    later.cqt[:, 31:] = np.random.default_rng(5).random((168, 29))
    p, q = train_v7_2.song_posteriors(model, song, a), train_v7_2.song_posteriors(model, later, a)
    np.testing.assert_allclose(p[:, :31 - a], q[:, :31 - a], atol=1e-6)
    assert not np.allclose(p[:, 31 - a], q[:, 31 - a])


@pytest.mark.parametrize("a", [0, 2])
def test_column_by_column_gives_the_whole_song_posteriors(a):
    torch.manual_seed(1)
    model = model_v6.ChordCRNN(bidirectional=False)
    song = random_song(n_frames=40)
    whole = train_v7_2.song_posteriors(model, song, a)
    s = LookaheadStreamingCRNN(model, a)
    out = [s.step(song.cqt[CENTRE_ROW:CENTRE_ROW + 144, t]) for t in range(40)]
    streamed = np.stack([p for p in out if p is not None] + s.flush(), axis=1)
    np.testing.assert_allclose(streamed, whole, atol=1e-5)


def one_note_song(roots, frames_per_note=10):
    targets = np.repeat(roots, frames_per_note)
    cqt = np.zeros((168, len(targets)), dtype=np.float32)
    for t, r in enumerate(targets):
        if r >= 0:
            cqt[12 + 2 * (24 + r), t] = 1.0
    return Song("toy", cqt, targets)


def test_training_with_lookahead_learns_toy_chords(monkeypatch):
    monkeypatch.setattr(train_v7_2, "device", lambda: torch.device("cpu"))
    cfg = {"train": {"seed": 0, "lr": 3e-3, "max_epochs": 15, "patience": 4, "dropout": 0.5,
                     "excerpt_frames": 16, "batch_excerpts": 4, "input_lookahead": 2}}
    train_songs = [one_note_song([0, 4, 7, 9, -1] * 2, frames_per_note=8)] * 8
    val = {"toy": [one_note_song([2, 5, 11], frames_per_note=8)]}
    model, history = train_v7_2.train(train_songs, val, cfg)
    assert max(h["val_toy"] for h in history) >= 0.9
    assert train_v7_2.frame_accuracy(model, val["toy"], 2) == max(h["val_toy"] for h in history)


def test_the_c2_grid_gets_its_own_cache():
    f0 = {"cqt": {"fmin": "F#0"}, "audio": {"harmonic_median_frames": 9}}
    c2 = {"cqt": {"fmin": "F#1"}, "audio": {"harmonic_median_frames": 9}}
    assert data_v7_2.cache_dir(f0).name == "v7_1_logcqt_median9"
    assert data_v7_2.cache_dir(c2).name == "v7_2_logcqt_Fs1"
