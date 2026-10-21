"""Tests for chordtracker.learned_emission (V4): behaviour, no pinned numbers."""
import numpy as np
import pytest

from chordtracker import learned_emission as le
from chordtracker.data.data import ChordAnnotation


def triad_beat(root: int, minor: bool, rng) -> np.ndarray:
    """A noisy (24,) beat: the triad in the treble, the root in the bass."""
    x = rng.uniform(0, 0.2, 24)
    for interval in (0, 3 if minor else 4, 7):
        x[(root + interval) % 12] = 1.0
    x[12 + root] = 1.0
    return x


def toy_take(rng, n_per_chord=6) -> le.BeatFeatures:
    """C:maj then A:min, one beat per 0.5 s."""
    X = np.stack([triad_beat(0, False, rng) for _ in range(n_per_chord)]
                 + [triad_beat(9, True, rng) for _ in range(n_per_chord)], axis=1)
    edges = np.arange(2 * n_per_chord + 1) * 0.5
    half = n_per_chord * 0.5
    reference = ChordAnnotation(np.array([[0.0, half], [half, 2 * half]]), ["C:maj", "A:min7"])
    return le.BeatFeatures("toy", X, edges, reference, 2 * half)


def test_beat_targets_take_the_longest_overlap_and_skip_unclassified():
    reference = ChordAnnotation(np.array([[0.0, 0.4], [0.4, 2.0], [2.0, 3.0]]), ["C:maj", "A:min7", "N"])
    targets = le.beat_targets(reference, np.array([0.0, 0.5, 1.0, 2.0, 3.0]))
    assert targets.tolist() == [0, 21, 21, -1]


def test_transpose_moves_roots_and_keeps_quality():
    rng = np.random.default_rng(0)
    X = np.stack([triad_beat(0, False, rng), triad_beat(9, True, rng)], axis=1)
    X_up, y_up = le.transpose(X, np.array([0, 21, -1])[:2], 2)
    assert y_up.tolist() == [2, 12 + 11]
    assert np.allclose(X_up[:, 0], np.concatenate([np.roll(X[:12, 0], 2), np.roll(X[12:, 0], 2)]))
    assert np.allclose(le.transpose(X_up, y_up, 10)[0], X)


def test_training_rows_cover_all_keys_with_balanced_roots():
    X, y = le.training_rows([toy_take(np.random.default_rng(1))])
    assert X.shape == (12 * 12, 24)
    counts = np.bincount(y, minlength=24)
    assert (counts[:12] == 6).all() and (counts[12:] == 6).all()


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(2)
    return le.fit(*le.training_rows([toy_take(rng) for _ in range(3)]), C=10.0)


def test_posteriors_are_distributions(model):
    P = le.posteriors(model, toy_take(np.random.default_rng(3)).features)
    assert P.shape == (24, 12)
    assert np.allclose(P.sum(axis=0), 1.0) and (P >= 0).all()
    assert np.isclose(model["prior"].sum(), 1.0)


def test_learned_model_recognises_unseen_keys(model):
    rng = np.random.default_rng(4)
    X = np.stack([triad_beat(r, m, rng) for m in (False, True) for r in range(12)], axis=1)
    assert le.posteriors(model, X).argmax(axis=0).tolist() == list(range(24))


def test_transposing_a_beat_transposes_its_posterior(model):
    x = triad_beat(4, False, np.random.default_rng(5))[:, None]
    P = le.posteriors(model, x)[:, 0]
    P_up = le.posteriors(model, le.transpose(x, np.array([4]), 3)[0])[:, 0]
    assert np.allclose(P_up[:12], np.roll(P[:12], 3), atol=0.05)
    assert np.allclose(P_up[12:], np.roll(P[12:], 3), atol=0.05)


def test_fit_needs_every_class():
    X, y = np.random.default_rng(6).uniform(size=(10, 24)), np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        le.fit(X, y, C=1.0)


def test_save_load_round_trip(model, tmp_path):
    le.save(model, tmp_path / "model.json")
    loaded = le.load(tmp_path / "model.json")
    X = toy_take(np.random.default_rng(7)).features
    assert np.allclose(le.posteriors(loaded, X), le.posteriors(model, X))
