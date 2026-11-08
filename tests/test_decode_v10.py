"""decode_v10: ensembles, quality offsets, the coordinate search and the level gate."""
import numpy as np
import pytest

from chordtracker import decode_v10, online_hmm


def test_ensemble_docstring_example():
    a, b = np.array([[0.9], [0.1]]), np.array([[0.5], [0.5]])
    assert np.allclose(decode_v10.ensemble([a, b], "arithmetic"), [[0.7], [0.3]])
    assert np.allclose(decode_v10.ensemble([a, b], "geometric"), [[0.75], [0.25]], atol=1e-6)
    with pytest.raises(ValueError):
        decode_v10.ensemble([a, b], "median")


def test_zero_offsets_change_nothing_and_offsets_act_per_quality():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(36), size=5).T                   # (36, 5): 3 qualities
    assert np.allclose(decode_v10.quality_offsets(p, np.zeros(3)), p)
    q = decode_v10.quality_offsets(p, np.array([0.0, 0.0, np.log(2)]))
    assert np.allclose(q.sum(axis=0), 1)
    assert np.allclose(q[24:] / q[:12].sum(axis=0, keepdims=True) * p[:12].sum(axis=0, keepdims=True), 2 * p[24:])


def test_coordinate_search_finds_the_grid_optimum_of_a_separable_objective():
    target = np.array([0.0, 0.5, -1.0])
    x, f, trace = decode_v10.coordinate_search(lambda o: -np.sum((o - target) ** 2), np.zeros(3),
                                               [-1.0, -0.5, 0.0, 0.5, 1.0], free=[1, 2])
    assert np.allclose(x, target) and f == 0 and trace[0]["objective"] == -1.25


def test_level_gate_puts_n_only_below_the_threshold():
    path = np.array([3, 3, 4, 5])
    assert decode_v10.level_gate(path, np.array([0.1, 0.5, 0.2, 0.9]), 0.3, 60).tolist() == [60, 3, 60, 5]


def test_decode_points_are_v8s_decoders():
    rng = np.random.default_rng(1)
    post = rng.dirichlet(np.ones(24), size=50).T
    A = np.full((24, 24), 0.2 / 23)
    np.fill_diagonal(A, 0.8)
    prior = np.full(24, 1 / 24)
    out = decode_v10.decode_points(post, {0.8: A}, prior, [2, 8])
    assert np.array_equal(out[("network", None)], post.argmax(axis=0))
    assert np.array_equal(out[("lag 8", 0.8)], online_hmm.fixed_lag_viterbi(post, A, prior, [8])[8])
