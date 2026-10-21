"""Tests for chordtracker.transitions (V3): behaviour, no pinned numbers."""
import librosa
import numpy as np
import pytest

from chordtracker import transitions as T
from chordtracker.templates import majmin_labels


@pytest.mark.parametrize("label, expected", [("D#:maj", (3, "maj")), ("A:min7", (9, "min")), ("G:7", (7, "maj")),
                                             ("B:hdim7", (11, "min")), ("N", None), ("E:sus4", None)])
def test_chord_class_reads_the_third(label, expected):
    assert T.chord_class(label) == expected


def test_change_counts_are_transposition_invariant():
    in_c = T.change_counts([["C:maj", "G:maj", "A:min"]])
    in_d = T.change_counts([["D:maj", "A:maj", "B:min"]])
    assert np.array_equal(in_c, in_d)
    assert in_c[0, 0, 7] == 1 and in_c[0, 1, 2] == 1 and in_c.sum() == 2


def test_change_counts_skip_repeats_same_class_and_unclassified_labels():
    counts = T.change_counts([["C:maj", "C:maj", "C:maj7", "N", "G:maj", "E:sus4", "A:min"]])
    assert counts.sum() == 0


def test_tau_zero_is_transition_loop():
    counts = T.change_counts([["C:maj", "G:maj", "C:maj", "F:maj"]])
    for p_self in [0.06, 0.5, 0.9]:
        A = T.transition(T.change_matrix(counts, alpha=1.0, tau=0.0), p_self)
        assert np.allclose(A, librosa.sequence.transition_loop(25, p_self))


def test_rows_sum_to_one_with_p_self_on_the_diagonal():
    A = T.transition(T.change_matrix(T.change_counts([["C:maj", "G:maj", "A:min", "F:maj"]])), 0.3)
    assert A.shape == (25, 25)
    assert np.allclose(A.sum(axis=1), 1.0)
    assert np.allclose(np.diag(A), 0.3)


def test_learned_moves_are_preferred_and_unseen_moves_keep_mass():
    Q = T.change_matrix(T.change_counts([["C:maj", "G:maj"]] * 10), alpha=1.0, tau=1.0)
    labels = majmin_labels()
    d, a, e_min = labels.index("D:maj"), labels.index("A:maj"), labels.index("E:min")
    assert Q[d, a] > Q[d, e_min] > 0          # maj -> maj up 7 was seen in another key; maj -> min up 2 never


def test_tau_flattens_the_prior():
    counts = T.change_counts([["C:maj", "G:maj"]] * 10)
    spread = [T.change_matrix(counts, alpha=1.0, tau=tau)[0].max() for tau in [0.0, 0.5, 1.0]]
    assert spread[0] < spread[1] < spread[2]
