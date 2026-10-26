"""V5.1 training: N survives the pitch shift, and the loop trains and reloads a 25-class model."""
import numpy as np
import torch

from chordtracker import train_v5, train_v5_1 as t
from chordtracker.data_v5 import Song


def toy_songs(n_frames: int = 40) -> list[Song]:
    """Two songs: silence labelled N, then a note on C (row 12 + 2 * 12 = C2) labelled C:maj."""
    songs = []
    for _ in range(2):
        cqt = np.zeros((168, n_frames), dtype=np.float32)
        cqt[36, n_frames // 2:] = 1.0
        y = np.full(n_frames, 24)
        y[n_frames // 2:] = 0
        songs.append(Song("toy", cqt, y))
    return songs


def test_make_batch_keeps_n_and_moves_chords():
    songs = toy_songs()
    _, y = t.make_batch(songs, np.array([[0, 0], [0, 30]]), np.array([3, 3]))
    assert y.tolist() == [24, 3]                         # N stays N, C:maj + 3 = D#:maj


def test_train_saves_and_loads_a_25_class_model(tmp_path):
    cfg = {"train": {"seed": 0, "lr": 1e-3, "batch": 16, "max_epochs": 2, "patience": 5, "dropout": 0.5}}
    model, history = t.train(toy_songs(), {"toy": toy_songs()}, cfg)
    assert len(history) == 2
    train_v5.save(model, tmp_path / "model.pt")
    again = t.load(tmp_path / "model.pt")
    assert train_v5.song_posteriors(again, toy_songs()[0]).shape == (25, 40)
