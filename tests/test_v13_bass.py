"""Behaviour of v13_bass: held chords, changes inside them, coincidence with bass moves, the per-frame
p_self decoder, the bass-move score and slash labels from the bass head."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v13_bass as v  # noqa: E402
from chordtracker import data_v8, online_hmm, transitions  # noqa: E402
from chordtracker.data.data import ChordAnnotation  # noqa: E402


def test_held_chords_join_inversions_and_drop_unscored_labels():
    a = ChordAnnotation(np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=float),
                        ["C:maj", "C:maj/3", "N", "G:7", "B:dim"])
    assert v.held_chords(a) == [(0.0, 2.0, 0), (3.0, 4.0, 31)]


def test_changes_inside_skip_the_ends_of_the_held_chord():
    hop_s = 0.1
    path = np.array([0] * 10 + [5] * 5 + [0] * 15 + [7] * 10)        # changes at 1.0, 1.5, 3.0 s
    held = [(0.0, 3.0, 0), (3.0, 4.0, 7)]
    got = v.changes_inside(path, held, hop_s, margin=0.3)
    assert [round(t, 2) for t, _, _ in got] == [1.0, 1.5]               # 3.0 s is the real change


def test_near_any_uses_the_window():
    assert v.near_any(np.array([1.0, 2.0, 5.0]), np.array([1.1, 4.0]), 0.15).tolist() == [True, False, False]
    assert v.near_any(np.array([1.0]), np.array([])).tolist() == [False]


def test_constant_p_self_gives_the_usual_fixed_lag_viterbi():
    rng = np.random.default_rng(0)
    post = rng.dirichlet(np.ones(5), size=30).T
    Q = rng.random((5, 5))
    np.fill_diagonal(Q, 0)
    Q /= Q.sum(axis=1, keepdims=True)
    prior = np.full(5, 0.2)
    for lag in (0, 4):
        expected = online_hmm.fixed_lag_viterbi(post, transitions.transition(Q, 0.9), prior, [lag])[lag]
        np.testing.assert_array_equal(v.tv_fixed_lag_viterbi(post, Q, np.full(30, 0.9), prior, lag), expected)


def test_the_bass_move_score_is_the_chance_two_draws_differ():
    q = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])                   # bass 0, 0, then 1
    np.testing.assert_allclose(v.bass_moves(q), [0.0, 0.0, 1.0])


def test_slash_labels_add_a_chord_tone_bass_only():
    names = data_v8.class_names("A")
    q = np.zeros((12, 3))
    q[4, 0] = q[0, 1] = q[2, 2] = 1.0                                   # E, C, D
    path = np.array([0, 0, 0])                                          # C:maj throughout
    assert v.slash_labels(path, q, names, half=0) == ["C:maj/3", "C:maj", "C:maj"]   # D is not a chord tone
    q7 = np.zeros((12, 1))
    q7[5, 0] = 1.0                                                      # F under G:7 is its b7
    assert v.slash_labels(np.array([31]), q7, names, half=0) == ["G:7/b7"]


def test_the_bass_models_chord_posteriors_are_the_usual_ones():
    import torch
    from chordtracker import model_v13
    from chordtracker.data_v5 import Song
    from chordtracker.train_v7_2 import song_posteriors
    torch.manual_seed(0)
    model = model_v13.ChordCRNNBass(60).eval()
    song = Song("x", np.random.default_rng(0).random((168, 50)).astype(np.float32), np.zeros(50, dtype=int))
    chord, bass = v.head_posteriors_v13(model, song, 2)
    np.testing.assert_allclose(chord, song_posteriors(model, song, 2), atol=1e-5)
    assert bass.shape == (13, 50)
    np.testing.assert_allclose(bass.sum(axis=0), 1, atol=1e-5)
