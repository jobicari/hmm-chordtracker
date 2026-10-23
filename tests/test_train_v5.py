"""Behaviour of V5 training: examples, batches, shift consistency, early stopping, save/load."""
import numpy as np
import torch

from chordtracker import train_v5 as tr
from chordtracker.data_v5 import Song


def one_note_song(roots, frames_per_note=10):
    """Toy song: each note is a 'major chord' target on its root, played as energy on the root's C3 row."""
    targets = np.repeat(roots, frames_per_note)
    cqt = np.zeros((168, len(targets)), dtype=np.float32)
    for t, r in enumerate(targets):
        if r >= 0:
            cqt[12 + 2 * (24 + r), t] = 1.0       # row 12 is C1, 24 semitones higher is C3
    return Song("toy", cqt, targets)


def test_training_frames_skip_frames_without_target():
    songs = [one_note_song([0, -1]), one_note_song([-1, 5, 7])]
    ex = tr.training_frames(songs)
    assert len(ex) == 30
    assert set(map(tuple, ex[:3])) == {(0, 0), (0, 1), (0, 2)}
    assert all(songs[i].targets[t] >= 0 for i, t in ex)


def test_a_batch_moves_input_and_target_together():
    """For every example and shift, the note in the patch is still the root of the shifted target."""
    songs = [one_note_song([0, 2, 9, 11])]
    rng = np.random.default_rng(0)
    ex = tr.training_frames(songs)
    shifts = rng.choice(list(range(-5, 7)), size=len(ex))
    x, y = tr.make_batch(songs, ex, shifts)
    assert x.shape == (40, 1, 144, 15) and y.dtype == torch.long
    note_row = x[:, 0, :, 7].argmax(dim=1)                       # centre frame
    assert torch.equal((note_row // 2) % 12, y % 12)


def test_epochs_since_best():
    assert tr.epochs_since_best([0.5, 0.7, 0.6, 0.65]) == 2
    assert tr.epochs_since_best([0.5, 0.6]) == 0


def test_training_learns_toy_chords_and_stops_early(monkeypatch):
    monkeypatch.setattr(tr, "device", lambda: torch.device("cpu"))
    train_songs = [one_note_song([0, 4, 7, 9] * 2, frames_per_note=30)]
    val = {"toy": [one_note_song([2, 5, 11])]}                    # roots never seen unshifted in train
    cfg = {"train": {"seed": 0, "lr": 3e-3, "batch": 32, "max_epochs": 15, "patience": 4, "dropout": 0.5}}
    model, history = tr.train(train_songs, val, cfg)
    assert max(h["val_toy"] for h in history) >= 0.9              # the shifts taught it the other roots
    assert len(history) < 15                                      # stopped once val stopped improving
    assert tr.frame_accuracy(model, val["toy"]) == max(h["val_toy"] for h in history)   # best epoch kept


def test_save_and_load_give_the_same_posteriors(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "device", lambda: torch.device("cpu"))
    torch.manual_seed(0)
    model = tr.ChordCNN().eval()
    tr.save(model, tmp_path / "model.pt")
    song = one_note_song([0, 7])
    assert np.allclose(tr.song_posteriors(model, song), tr.song_posteriors(tr.load(tmp_path / "model.pt"), song))
