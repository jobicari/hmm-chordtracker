"""Behaviour of the V8 prior and transition matrix: every key equally likely, rows that sum to 1, moves
split by quality share, V3's matrix back when the prior holds only maj and min, and moves counted
from training targets."""
import numpy as np
import pytest

from chordtracker import transitions, transitions_v8


def billboard_like(seed=0):
    return np.random.default_rng(seed).integers(0, 500, size=(2, 2, 12)).astype(float)


def test_prior_counts_every_target_in_all_12_keys():
    prior = transitions_v8.training_prior(np.array([0, 0, 36, -1]), 60)
    assert prior[:12] == pytest.approx(np.full(12, 2 / 36))
    assert prior[36:48] == pytest.approx(np.full(12, 1 / 36))
    assert prior[12:36].sum() == 0 and prior[48:].sum() == 0


@pytest.mark.parametrize("vocab", ["A", "B", "B9"])
def test_rows_sum_to_one_with_a_zero_diagonal(vocab):
    n = 12 * len(transitions_v8.VOCABULARIES[vocab])
    prior = np.random.default_rng(1).random(n)
    Q = transitions_v8.change_matrix(billboard_like(), vocab, prior / prior.sum())
    assert Q.shape == (n, n)
    np.testing.assert_allclose(Q.sum(axis=1), 1.0)
    assert np.all(np.diag(Q) == 0) and np.all(Q >= 0)


def test_with_only_maj_and_min_it_is_v3s_chord_block():
    counts = billboard_like()
    prior = np.zeros(60)
    prior[:24] = 1 / 24
    Q = transitions_v8.change_matrix(counts, "A", prior)
    v3 = transitions.change_matrix(counts)[:24, :24]
    np.testing.assert_allclose(Q[:24, :24], v3 / v3.sum(axis=1, keepdims=True))
    assert Q[:, 24:].sum() == 0


def test_a_move_is_split_by_the_quality_share_in_its_family():
    """C:maj -> G goes to G:maj and G:7 as the prior holds maj and 7 (3 : 1); G:min7 gets the min share."""
    prior = np.zeros(60)
    prior[0:12], prior[24:36] = 3, 1           # maj, 7
    prior[12:24], prior[48:60] = 1, 1          # min, min7
    Q = transitions_v8.change_matrix(billboard_like(), "A", prior / prior.sum())
    assert Q[0, 7] / Q[0, 24 + 7] == pytest.approx(3.0)
    assert Q[0, 12 + 7] / Q[0, 48 + 7] == pytest.approx(1.0)
    assert Q[0, 36 + 7] == 0                   # maj7 has no prior mass


def test_same_root_quality_change_gets_alpha_only():
    counts = billboard_like()
    counts[0, 0, 0] = 0
    prior = np.full(60, 1 / 60)
    Q = transitions_v8.change_matrix(counts, "A", prior, alpha=1.0)
    raw_same = 1.0 * (1 / 3)                   # alpha, times 7's share of the maj family (maj, 7, maj7)
    raw_fifth = (counts[0, 0, 7] + 1.0) * (1 / 3)
    assert Q[0, 24] / Q[0, 24 + 7] == pytest.approx(raw_same / raw_fifth)


def test_label_counts_count_changes_by_quality_and_interval():
    counts = transitions_v8.label_counts([np.array([0, 0, 24, 24, 7]), np.array([12, -1, 12, 5])], 5)
    assert counts[0, 2, 0] == 1                # C:maj -> C:7, same root
    assert counts[2, 0, 7] == 1                # C:7 -> G:maj, up a fifth
    assert counts[1, 0, 5] == 1                # C:min -> F:maj; the -1 gap between two C:min is no change
    assert counts.sum() == 3


@pytest.mark.parametrize("n_q", [2, 5, 8])
def test_label_change_matrix_rows_sum_to_one_with_a_zero_diagonal(n_q):
    counts = np.random.default_rng(2).integers(0, 50, size=(n_q, n_q, 12)).astype(float)
    Q = transitions_v8.label_change_matrix(counts)
    assert Q.shape == (12 * n_q, 12 * n_q)
    np.testing.assert_allclose(Q.sum(axis=1), 1.0)
    assert np.all(np.diag(Q) == 0) and np.all(Q[~np.eye(len(Q), dtype=bool)] > 0)


def test_label_change_matrix_follows_the_counts_in_every_key():
    """C:maj -> C:7 counted 9 times and C:maj -> G:maj 4 times: 10 : 5 with alpha 1, from D as from C."""
    counts = np.zeros((5, 5, 12))
    counts[0, 2, 0], counts[0, 0, 7] = 9, 4
    Q = transitions_v8.label_change_matrix(counts, alpha=1.0)
    assert Q[0, 24] / Q[0, 7] == pytest.approx(2.0)
    assert Q[2, 26] / Q[2, 9] == pytest.approx(2.0)
    assert Q[0, 12] / Q[0, 7] == pytest.approx(1 / 5)   # a move never seen: alpha only


def test_tempered_prior_goes_from_uniform_to_the_prior():
    prior = np.array([0.75, 0.25])
    np.testing.assert_allclose(transitions_v8.tempered_prior(prior, 1.0), prior)
    np.testing.assert_allclose(transitions_v8.tempered_prior(prior, 0.0), [0.5, 0.5])
    half = transitions_v8.tempered_prior(prior, 0.5)
    assert half.sum() == pytest.approx(1.0) and 0.5 < half[0] < 0.75
