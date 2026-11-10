"""Behaviour of V15's vocabulary A+: targets of the new qualities, the pitch shift with N in the middle of the
model order, the decode order, the prior and matrix over the new qualities, the extended model that starts
as V14, the time-stretched focus excerpts and the per-quality scores."""
import numpy as np
import pytest
import torch

from chordtracker import (data_v5, data_v6, data_v7_2, data_v8, data_v15, eval_v15, frame_score, model_v14,
                          train_v15, transitions_v8)
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5 import Song

V = data_v15.Vocab()                     # dim, aug, hdim7; dim7 -> dim
N = 60


# ---------------------------------------------------------------- targets

@pytest.mark.parametrize("label, cls", [
    ("C:maj", 0), ("A:min7", 57), ("Bb:7/3", 34), ("N", N), ("X", -1), ("D:sus4", -1), ("C:minmaj7", -1),
    ("C:dim", 61), ("B:dim", 72), ("C:dim/b3", 61), ("E:dim(9)", 65), ("D:dim7", 63),
    ("C:aug", 73), ("C#:aug/3", 74), ("C:aug(b7)", 73),
    ("C:hdim7", 85), ("B:hdim7", 96), ("C:min7(b5)", 85), ("C:dim(b7)", 85), ("C:min(b5)", -1),
])
def test_targets_of_vocabulary_a_plus(label, cls):
    assert V.target(label) == cls


def test_vocabulary_a_and_n_keep_v14s_indices():
    for label in ("C:maj", "F#:min", "G:7", "Eb:maj7", "B:min7", "C:maj6", "N", "X", "C:sus2"):
        assert V.target(label) == data_v8.target(label, "A", with_n=True)
    assert V.class_names()[:61] == data_v8.class_names("A", with_n=True)
    assert V.class_names()[61] == "C:dim" and V.class_names()[-1] == "B:hdim7" and V.n_classes == 97


def test_dim7_gets_its_own_block_when_it_is_a_new_quality():
    v4 = data_v15.Vocab(("dim", "aug", "dim7", "hdim7"))
    assert v4.n_classes == 109
    assert v4.target("D:dim7") == 61 + 24 + 2 and v4.target("C:dim") == 61 and v4.target("C:hdim7") == 97
    unmerged = data_v15.Vocab(("dim",), merge={})
    assert unmerged.target("D:dim7") == -1 and unmerged.target("C:aug") == -1 and unmerged.target("C:dim") == 61
    with pytest.raises(ValueError):
        data_v15.Vocab(("sus4",))


def test_frame_targets_read_the_annotation():
    ann = ChordAnnotation(np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]), ["N", "G:hdim7", "X"])
    assert V.frame_targets(ann, np.array([0.5, 1.5, 2.5, 3.5])).tolist() == [N, 85 + 7, -1, -1]


# ---------------------------------------------------------------- shift and decode order

@pytest.mark.parametrize("k", list(data_v5.SHIFTS))
def test_the_shift_moves_every_root_within_its_block_and_leaves_n(k):
    y = np.arange(-1, V.n_classes)
    s = V.shift_targets(y, k)
    assert s[0] == -1 and s[1 + N] == N
    a = y[(y >= 0) & (y < N)]
    assert V.shift_targets(a, k).tolist() == data_v5.shift_targets(a, k).tolist()     # A as V14
    new = y[y > N]
    moved = V.shift_targets(new, k)
    assert ((moved - 61) // 12).tolist() == ((new - 61) // 12).tolist()              # same quality
    assert ((moved - 61) % 12).tolist() == (((new - 61) % 12 + k) % 12).tolist()     # root up k
    assert V.shift_targets(V.shift_targets(y, k), -k).tolist() == y.tolist()


def test_the_decode_order_puts_n_last():
    order = V.decode_order
    assert sorted(order.tolist()) == list(range(97)) and order[-1] == N and order[60] == 61
    assert V.decode_names()[-1] == "N" and V.decode_names()[60] == "C:dim"
    assert V.to_decode(np.array([0, 59, N, 61, 96, -1])).tolist() == [0, 59, 96, 60, 95, -1]
    assert np.array_equal(V.to_decode(order), np.arange(97))
    assert V.qualities == ("maj", "min", "7", "maj7", "min7", "dim", "aug", "hdim7")


def test_shifted_excerpts_through_make_batch():
    targets = np.array([0, N, 61, 96, -1, 73])
    song = Song("x", np.random.default_rng(0).random((168, 6)).astype(np.float32), targets)
    _, y = train_v15.make_batch([song], np.array([[0, 0]]), np.array([2]), np.array([1.0]), 6, 2, V)
    assert y[0].tolist() == [2, N, 63, 86, -1, 75]


# ---------------------------------------------------------------- prior and matrix

def test_the_prior_and_the_matrix_over_the_new_qualities():
    y = V.to_decode(np.array([0, 0, 61, N]))                                    # C:maj twice, C:dim, N
    prior = transitions_v8.training_prior(y, V.n_classes, V.n_chords)
    assert np.isclose(prior[:12].sum(), 0.5) and np.isclose(prior[60:72].sum(), 0.25) and np.isclose(prior[-1], 0.25)
    assert np.allclose(prior[60:72], prior[60])                                 # every root of dim alike
    counts = np.ones((2, 2, 12))
    Q = transitions_v8.change_matrix(counts, V.qualities, prior + 1e-3, 1.0, 1.0, chord_to_n=0.02)
    assert Q.shape == (97, 97) and np.allclose(Q.sum(axis=1), 1) and np.allclose(Q[:-1, -1], 0.02)
    # with vocabulary A's qualities it is V14's matrix
    p61 = np.random.default_rng(0).random(61)
    assert np.allclose(transitions_v8.change_matrix(counts, data_v15.BASE, p61, chord_to_n=0.01),
                       transitions_v8.change_matrix(counts, "A", p61, chord_to_n=0.01))
    assert transitions_v8.families(V.qualities).tolist() == [0, 1, 0, 0, 1, 1, 0, 1]


def test_a_class_without_prior_mass_does_not_take_over_the_decoder():
    rng = np.random.default_rng(0)
    post = rng.dirichlet(np.ones(97), size=50).T
    post[0] += 2.0
    post /= post.sum(axis=0)
    prior = np.full(97, 1 / 85)
    prior[72:84] = 0.0                                                           # no aug in the train labels
    A, d = eval_v15.decoder(V.qualities, np.ones((2, 2, 12)), prior, prior, 0.98, 0.01, 1.0, 0.25)
    assert np.isfinite(d).all() and (eval_v15.decode(post, A, d, 8) == 0).mean() > 0.9


# ---------------------------------------------------------------- the extended model

def test_the_extended_model_keeps_v14s_outputs_and_starts_the_new_rows_quiet():
    torch.manual_seed(0)
    base = model_v14.ChordCRNN(61, dropout=0.5).eval()
    prior = np.full(97, 1 / 97)
    prior[61:] /= 40.0
    prior /= prior.sum()
    ext = train_v15.extend(base, V, prior).eval()
    x = torch.randn(2, 1, 144, 30)
    with torch.no_grad():
        old, new = base(x), ext(x)
    assert torch.allclose(new[..., :61], old, atol=1e-6)
    offset = float(np.log(40.0))
    assert torch.allclose(new[..., 61:73], old[..., 12:24] - offset, atol=1e-5)          # dim from min
    assert torch.allclose(new[..., 73:85], old[..., 0:12] - offset, atol=1e-5)           # aug from maj
    assert torch.allclose(new[..., 85:97], old[..., 48:60] - offset, atol=1e-5)          # hdim7 from min7


# ---------------------------------------------------------------- focus excerpts and the stretch

def test_rate_one_is_the_v14_excerpt():
    rng = np.random.default_rng(1)
    song = Song("x", rng.random((168, 40)).astype(np.float32), rng.integers(-1, 97, 40))
    for start, k in ((-5, 0), (3, 2), (30, -5)):
        x, y = train_v15.stretched_excerpt(song, start, k, 16, 2, 1.0)
        assert np.allclose(x, data_v7_2.excerpt(song, start, k, 16, 2))
        assert y.tolist() == data_v6.excerpt_targets(song, start, 0, 16).tolist()


def test_a_stretched_excerpt_reads_the_song_at_its_rate():
    cqt = np.tile(np.arange(100, dtype=np.float32), (168, 1))                 # column t holds the value t
    song = Song("x", cqt, np.arange(100) % 97)
    x, y = train_v15.stretched_excerpt(song, 10, 0, 20, 2, 1.05)
    src = 10 + (np.arange(34) - 14 + 2) * 1.05
    assert np.allclose(x[0], np.where(src < 0, 0.0, src), atol=1e-4)        # linear in time at 1.05, zeros before
    assert y.tolist() == (np.rint(10 + np.arange(20) * 1.05).astype(int) % 97).tolist()


def test_focus_excerpts_hold_their_run():
    y = np.full(400, 0)
    y[50:60], y[300:310], y[320:330] = 61, 75, 90
    songs = [Song("a", np.zeros((168, 400), dtype=np.float32), y)]
    segs = train_v15.focus_segments(songs, V)
    assert segs == [(0, 50, 59, 0), (0, 300, 309, 1), (0, 320, 329, 2)]
    starts = train_v15.focus_starts(segs, 300, 128, np.random.default_rng(0))
    for _, s in starts:
        held = [first >= s and last < s + 128 for _, first, last, _ in segs]
        assert any(held)


# ---------------------------------------------------------------- scores

def test_tetrads_tables_and_the_per_quality_scores():
    ann = ChordAnnotation(np.array([[0.0, 0.95], [0.95, 2.0]]), ["C:dim", "C:maj"])     # change on a cell edge
    t = eval_v15.song_tables(ann, 20, V, 0.1, ("sevenths", "tetrads"))
    names = V.decode_names()
    path = np.array([names.index("C:dim")] * 10 + [0] * 10)
    assert frame_score.score_path(t["tetrads_M"], t["tetrads_S"], path) == pytest.approx((2.0, 2.0))
    assert frame_score.score_path(t["sevenths_M"], t["sevenths_S"], path) == pytest.approx((1.05, 1.05))  # dim left out
    targets = V.to_decode(V.frame_targets(ann, np.arange(20) * 0.1))
    songs = {"s": {"set": "rwc", "targets": targets, **t}}
    half = path.copy()
    half[5:10] = names.index("C:min")
    r = eval_v15.score({"s": half}, songs, V, 0.1, ("sevenths", "tetrads"))
    assert r["quality"]["dim"]["recall"] == pytest.approx(50.0) and r["quality"]["dim"]["precision"] == 100.0
    assert r["quality"]["min"]["precision"] == 0.0 and r["sets"]["rwc"]["tetrads"] == pytest.approx(75.0)
