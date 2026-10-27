"""Behaviour of V6 training: it learns toy chords in every key, stops early, keeps the best epoch, saves and loads."""
import numpy as np
import torch

from chordtracker import train_v6 as tr
from chordtracker.data_v5 import Song


def one_note_song(roots, frames_per_note=10):
    """Toy song: each note is a 'major chord' target on its root, played as energy on the root's C3 row."""
    targets = np.repeat(roots, frames_per_note)
    cqt = np.zeros((168, len(targets)), dtype=np.float32)
    for t, r in enumerate(targets):
        if r >= 0:
            cqt[12 + 2 * (24 + r), t] = 1.0       # row 12 is C1, 24 semitones higher is C3
    return Song("toy", cqt, targets)


CFG = {"train": {"seed": 0, "lr": 3e-3, "max_epochs": 15, "patience": 4, "dropout": 0.5,
                 "excerpt_frames": 16, "batch_excerpts": 4, "bidirectional": True}}


def test_training_learns_toy_chords_and_stops_early(monkeypatch):
    monkeypatch.setattr(tr, "device", lambda: torch.device("cpu"))
    train_songs = [one_note_song([0, 4, 7, 9, -1] * 2, frames_per_note=8)] * 8     # 8 copies: ~40 excerpts per epoch
    val = {"toy": [one_note_song([2, 5, 11], frames_per_note=8)]}    # roots never seen unshifted in train
    model, history = tr.train(train_songs, val, CFG)
    assert max(h["val_toy"] for h in history) >= 0.9              # the shifts taught it the other roots
    assert len(history) < 15                                      # stopped once val stopped improving
    assert tr.frame_accuracy(model, val["toy"]) == max(h["val_toy"] for h in history)   # best epoch kept
    assert all(h["train_loss"] > 0 and h["seconds"] > 0 for h in history)


def test_song_posteriors_cover_every_frame():
    torch.manual_seed(0)
    P = tr.song_posteriors(tr.ChordCRNN(), one_note_song([0, 7, -1], frames_per_note=5))
    assert P.shape == (24, 15)
    assert np.allclose(P.sum(axis=0), 1, atol=1e-5)


def test_save_and_load_give_the_same_posteriors(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "device", lambda: torch.device("cpu"))
    torch.manual_seed(0)
    song = one_note_song([0, 7])
    for bidirectional in (True, False):
        model = tr.ChordCRNN(bidirectional=bidirectional).eval()
        tr.save(model, tmp_path / "model.pt")
        loaded = tr.load(tmp_path / "model.pt", bidirectional=bidirectional)
        assert np.allclose(tr.song_posteriors(model, song), tr.song_posteriors(loaded, song))
