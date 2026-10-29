"""Behaviour of the hold rule: it removes short runs, keeps long ones, and never reads past t + m - 1."""
import numpy as np

from chordtracker.hold import hold


def test_m_1_is_the_network_alone():
    a = np.array([0, 1, 0, 2, 2, 3])
    assert np.array_equal(hold(a, 1), a)


def test_runs_shorter_than_m_are_held_over():
    a = np.array([0, 0, 1, 0, 0, 2, 2, 2, 3, 2, 2])
    assert list(hold(a, 2)) == [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2]
    assert list(hold(a, 3)) == [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2]


def test_a_label_never_reads_past_its_lookahead():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, 60)
    b = a.copy()
    b[40:] = rng.integers(0, 3, 20)
    for m in (2, 3, 5):
        assert np.array_equal(hold(a, m)[:40 - m + 1], hold(b, m)[:40 - m + 1])
