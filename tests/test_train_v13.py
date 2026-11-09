"""Behaviour of V13's played-bass training: frame targets from MIDI notes, the pitch shift moving the bass
with the audio, the bass model's chord output equal to V8's, and one training step on a toy song."""
import numpy as np
import pytest
import torch

from chordtracker import data_v5, data_v7_2, model_v8, model_v13, train_v13
from chordtracker.data_v5 import CENTRE_ROW, Song

SR, HOP = 22050, 2048


def test_played_bass_is_the_lowest_sounding_note_or_none():
    notes = [(0.0, 0.5, 40, 80), (0.4, 1.0, 45, 80), (0.42, 0.48, 52, 80)]
    times = np.array([0.0, 0.45, 0.9, 1.2])
    frames = times * SR / HOP
    bass = train_v13.played_bass(notes, 14, SR, HOP)
    got = bass[np.round(frames).astype(int)]
    t = np.round(frames).astype(int) * HOP / SR                           # the frames' own times
    assert t[1] < 0.5 and got.tolist() == [4, 4, 9, 12]


def test_the_shift_moves_the_bass_like_the_chord_root():
    bass = np.array([0, 4, 12, 9, 11])
    y = train_v13.excerpt_bass(bass, 0, 3, 7)
    assert y.tolist() == [3, 7, 12, 0, 2, -1, -1]                         # "no bass" stays, outside is -1
    assert train_v13.excerpt_bass(None, 0, 3, 4).tolist() == [-1] * 4
    roots = data_v5.shift_targets(np.array([0, 4, 9, 11]), 3)             # C, E, A, B major -> +3
    assert (roots % 12).tolist() == [3, 7, 0, 2]


@pytest.mark.parametrize("k", [-5, 0, 3, 6])
def test_the_shifted_bass_target_is_the_pitch_the_network_sees(k):
    """A song whose only energy is the bin of pitch class 7 (G): after a shift of k, the strip's bright
    row reads as pitch class 7 + k, which is the bass target excerpt_bass gives."""
    cqt = np.zeros((168, 40), dtype=np.float32)
    g = CENTRE_ROW + 2 * (7 + 12)                                          # G2: 24 bins an octave, C1 at CENTRE_ROW
    cqt[g] = 1.0
    song = Song("x", cqt, np.zeros(40, dtype=int))
    strip = data_v7_2.excerpt(song, 10, k, 8, 2)
    row = int(strip[:, 10].argmax())
    seen = (row // 2) % 12
    assert seen == train_v13.excerpt_bass(np.full(40, 7), 10, k, 8)[0]


def test_the_bass_model_keeps_the_chord_models_output():
    torch.manual_seed(0)
    chord = model_v8.ChordCRNN(60, bidirectional=False).eval()
    model = model_v13.from_chord_model(chord.state_dict(), 60, 0.5).eval()
    x = torch.rand(2, 1, 144, 30)
    logits, bass = model.heads(x)
    torch.testing.assert_close(logits, chord(x))
    torch.testing.assert_close(model(x), chord(x))
    assert bass.shape == (2, 16, 13)
    with pytest.raises(ValueError):
        model_v13.from_chord_model({"nonsense": torch.zeros(1)}, 60, 0.5)


def test_one_epoch_learns_from_the_bass_and_masks_songs_without_it():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    songs = [Song(f"s{i}", rng.random((168, 140)).astype(np.float32), rng.integers(0, 60, 140)) for i in range(2)]
    bass = [rng.integers(0, 13, 140), None]
    t = {"excerpt_frames": 64, "batch_excerpts": 2, "input_lookahead": 2, "kl_weight": 0, "bass_weight": 0.5}
    model = model_v13.ChordCRNNBass(60)
    before = model.bass.weight.detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce, kl = train_v13.run_epoch(model, opt, songs, np.random.default_rng(0), t, None, bass)
    assert np.isfinite(ce) and kl == 0.0
    assert not torch.equal(model.bass.weight, before)
