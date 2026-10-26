"""V5.1's 25-state prior, and that V5's pooling and decoder take 25 states as they are."""
import numpy as np

from chordtracker import acoustic_hmm, acoustic_hmm_v5_1 as ah, transitions
from chordtracker.data.data import ChordAnnotation


def test_prior_keeps_the_major_minor_balance_and_ns_share():
    prior = ah.training_prior(np.array([0, 0, 12, 24, -1]))
    assert np.isclose(prior.sum(), 1)
    assert np.allclose(prior[:12], 2 / 48) and np.allclose(prior[12:24], 1 / 48)
    assert np.isclose(prior[24], 1 / 4)                  # 1 of the 4 frames with a target is N


def test_without_n_it_is_v5s_prior():
    y = np.array([0, 3, 15, 15, 21, -1])
    prior = ah.training_prior(y)
    assert prior[24] == 0 and np.allclose(prior[:24], acoustic_hmm.training_prior(y))


def test_frames_to_beats_pools_25_rows():
    P = np.full((25, 4), 1 / 25)
    P[:, :2] = 0.01
    P[24, :2] = 1 - 24 * 0.01                            # two N frames, then two flat frames
    beats = acoustic_hmm.frames_to_beats(P, np.array([0.05, 0.15, 0.25, 0.35]), np.array([0.0, 0.2, 0.4]))
    assert beats.shape == (25, 2)
    assert beats[:, 0].argmax() == 24


def test_decode_runs_on_25_states_and_finds_n():
    Q = transitions.change_matrix(np.ones((2, 2, 12)))   # (25, 25), N row and column included
    beats = np.full((25, 6), 0.01)
    beats[24, :2] = beats[7, 2:] = 0.76                  # N, N, then G:maj
    path = acoustic_hmm.decode(beats / beats.sum(axis=0), Q, p_self=0.5, prior=np.full(25, 1 / 25))
    assert list(path) == [24, 24, 7, 7, 7, 7]


def test_labels_end_with_n():
    assert len(ah.LABELS) == 25 and ah.LABELS[0] == "C:maj" and ah.LABELS[24] == "N"


def test_change_prior_keeps_v5s_chord_moves_and_adds_n():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 500, size=(2, 2, 12)).astype(float)
    Q = ah.change_prior(counts)
    assert Q.shape == (25, 25) and np.allclose(Q.sum(axis=1), 1) and np.allclose(np.diag(Q), 0)
    assert np.allclose(Q[:24, 24], 1 / 24) and np.allclose(Q[24, :24], 1 / 24)
    v5 = transitions.change_matrix(counts)[:24, :24]
    v5 = v5 / v5.sum(axis=1, keepdims=True)          # V5's Q (run_v5.change_prior)
    assert np.allclose(Q[:24, :24], v5 * 23 / 24)


def test_n_seconds_splits_found_missed_and_false():
    ref = ChordAnnotation(np.array([[0.0, 2.0], [2.0, 6.0], [6.0, 7.0]]), ["N", "C:maj", "N"])
    est = ChordAnnotation(np.array([[0.0, 3.0], [3.0, 7.0]]), ["N", "C:maj"])
    n = ah.n_seconds(ref, est)
    assert np.isclose(n["ref_n"], 3) and np.isclose(n["found"], 2)
    assert np.isclose(n["missed"], 1) and np.isclose(n["false_n"], 1)
