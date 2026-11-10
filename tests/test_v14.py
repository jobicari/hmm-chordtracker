"""Behaviour of V14's N class and level input: N as a target that survives the pitch shift, the prior and the
matrices with an N row and column, mir_eval scoring an N output, the dBFS level offline and live, and the
model with the level read the same way on a whole song and one column at a time."""
import mir_eval
import numpy as np
import pytest
import torch

from chordtracker import (data_v5, data_v6, data_v7_2, data_v8, evaluate, features_v14, model_v8, model_v14,
                          train_v14, transitions_v8)
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5 import Song
from chordtracker.train_v14 import LevelSong

N = 60                           # N in vocabulary A


# ---------------------------------------------------------------- targets

def test_n_is_a_class_only_with_with_n():
    assert data_v8.target("N", "A", with_n=True) == N
    assert data_v8.target("N", "A") == -1                                 # V8 unchanged
    assert data_v8.target("X", "A", with_n=True) == -1
    assert data_v8.target("D:sus4", "A", with_n=True) == -1               # out of vocabulary stays -1
    assert data_v8.target("G:7", "A", with_n=True) == 31
    assert data_v8.target("N", "B", with_n=True) == 84
    names = data_v8.class_names("A", with_n=True)
    assert len(names) == 61 and names[N] == "N" and names[:60] == data_v8.class_names("A")


def test_frame_targets_with_n_leave_frames_outside_the_annotation_at_minus_one():
    ann = ChordAnnotation(np.array([[0.5, 1.0], [1.0, 2.0], [2.0, 3.0]]), ["N", "C:maj", "X"])
    times = np.array([0.2, 0.7, 1.5, 2.5, 3.5])
    assert data_v8.frame_targets(ann, times, "A", with_n=True).tolist() == [-1, N, 0, -1, -1]
    assert data_v8.frame_targets(ann, times, "A").tolist() == [-1, -1, 0, -1, -1]


@pytest.mark.parametrize("k", list(data_v5.SHIFTS))
def test_n_survives_every_pitch_shift(k):
    y = np.array([0, 59, N, -1, 21])
    shifted = data_v5.shift_targets(y, k, n_chords=N)
    assert shifted[2] == N and shifted[3] == -1
    assert shifted[[0, 1, 4]].tolist() == data_v5.shift_targets(y[[0, 1, 4]], k).tolist()
    song = Song("x", np.zeros((168, 5), dtype=np.float32), y)
    assert data_v6.excerpt_targets(song, 0, k, 5, n_chords=N)[2] == N
    _, batch = data_v7_2.make_batch([song], np.array([[0, 0]]), np.array([k]), 5, 2, n_chords=N)
    assert batch[0, 2].item() == N
    level_song = LevelSong("x", song.cqt, y, np.zeros(5, dtype=np.float32))
    _, batch, _ = train_v14.make_batch([level_song], np.array([[0, 0]]), np.array([k]), 5, 2, N, True)
    assert batch[0, 2].item() == N


def test_the_old_shift_is_unchanged_without_n_chords():
    assert data_v5.shift_targets(np.array([0, 23, 59, -1]), 2).tolist() == [2, 13, 49, -1]
    assert data_v5.shift_targets(np.array([60]), 2).tolist() == [62]          # why N needs n_chords


# ---------------------------------------------------------------- prior and matrices

def test_the_prior_counts_n_unshifted_at_its_frame_share():
    prior = transitions_v8.training_prior(np.array([0, 0, 36, N, -1]), 61, n_chords=N)
    assert prior.sum() == pytest.approx(1.0)
    assert prior[N] == pytest.approx(1 / 4)
    assert prior[0] == pytest.approx(prior[5]) == pytest.approx(2 / 4 / 12)
    assert prior[36] == pytest.approx(1 / 4 / 12)
    np.testing.assert_allclose(transitions_v8.training_prior(np.array([0, 0, 36]), 60),
                               transitions_v8.training_prior(np.array([0, 0, 36]), 60, n_chords=60))


def test_n_change_counts_and_the_rate():
    counts = transitions_v8.n_change_counts([np.array([0, 0, N, N, 7, -1, N, 7, 9])], N)
    assert counts == {"chord_to_chord": 1, "chord_to_n": 1, "n_to_chord": 2}
    assert transitions_v8.n_rate(counts) == pytest.approx(0.5)
    # label_counts reads N as no chord: of [C:maj, N, G:maj, A:maj] only G:maj -> A:maj counts
    assert transitions_v8.label_counts([np.array([0, N, 7, 9])], 5).sum() == 1


def _prior():
    rng = np.random.default_rng(0)
    p = rng.random(61)
    return p / p.sum()


@pytest.mark.parametrize("kind", ["billboard", "train"])
def test_the_matrices_with_n_are_stochastic_with_a_zero_diagonal(kind):
    prior, rate = _prior(), 0.03
    if kind == "billboard":
        counts = np.random.default_rng(1).integers(0, 50, (2, 2, 12)).astype(float)
        Q = transitions_v8.change_matrix(counts, "A", prior, 1.0, 1.0, chord_to_n=rate)
        Q0 = transitions_v8.change_matrix(counts, "A", prior[:60], 1.0, 1.0)
    else:
        counts = np.random.default_rng(1).integers(0, 50, (5, 5, 12)).astype(float)
        Q = transitions_v8.label_change_matrix(counts, 1.0, 1.0, prior=prior, chord_to_n=rate)
        Q0 = transitions_v8.label_change_matrix(counts, 1.0, 1.0)
    assert Q.shape == (61, 61)
    np.testing.assert_allclose(Q.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.diag(Q), 0.0)
    np.testing.assert_allclose(Q[:60, N], rate)
    np.testing.assert_allclose(Q[:60, :60], (1 - rate) * Q0)
    np.testing.assert_allclose(Q[N, :60], prior[:60] / prior[:60].sum())


# ---------------------------------------------------------------- scoring

@pytest.mark.parametrize("metric", ["majmin", "sevenths"])
def test_mir_eval_scores_an_n_output_against_a_reference_n(metric):
    ref = ChordAnnotation(np.array([[0.0, 1.0], [1.0, 2.0]]), ["N", "C:maj"])
    labels = data_v8.class_names("A", with_n=True)
    times = (np.arange(20) + 0.5) * 0.1                                     # the change falls on 1.0 s
    right = evaluate.path_to_annotation(np.array([N] * 10 + [0] * 10), labels, times, 2.0)
    wrong = evaluate.path_to_annotation(np.array([0] * 20), labels, times, 2.0)
    assert evaluate.score(ref, right, metric).wcsr == pytest.approx(1.0)
    assert evaluate.score(ref, wrong, metric).wcsr == pytest.approx(0.5)
    assert mir_eval.chord.sevenths(["N"], ["N"])[0] == 1


# ---------------------------------------------------------------- level

def test_window_level_examples():
    t = np.arange(2048)
    assert features_v14.window_level(np.ones(2048)) == pytest.approx(0.0)
    assert features_v14.window_level(np.sin(2 * np.pi * 16 * t / 2048)) == pytest.approx(-0.0301, abs=1e-4)
    assert features_v14.window_level(np.zeros(2048)) == -1.0
    assert features_v14.window_level(np.full(2048, 1e-9)) == -1.0              # -180 dB clipped


@pytest.mark.parametrize("block", [2048, 1000, 4096, 333])
def test_offline_and_live_levels_agree(block):
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(2048 * 9 + 777) * np.linspace(0, 0.5, 2048 * 9 + 777)).astype(np.float32)
    y[5000:9000] = 0.0
    offline = features_v14.frame_levels(y, 2048)
    live = features_v14.StreamingLevels(2048)
    got = [v for i in range(0, len(y), block) for v in live.step(y[i:i + block])]
    assert len(offline) == 1 + len(y) // 2048 == len(got)
    np.testing.assert_array_equal(np.array(got, dtype=np.float32), offline)
    assert offline[0] == -1.0                                                 # frame 0 hears only the past


def test_the_level_of_frame_t_reads_only_the_audio_before_it():
    y = np.zeros(2048 * 6, dtype=np.float32)
    y[2048 * 3:] = 0.5                                                        # sound from frame 3's centre on
    levels = features_v14.frame_levels(y, 2048)
    assert levels[:4].tolist() == [-1.0] * 4 and levels[4] > -0.1


# ---------------------------------------------------------------- model

def test_without_the_level_the_model_is_v8s():
    torch.manual_seed(3)
    v8 = model_v8.ChordCRNN(61, bidirectional=False)
    torch.manual_seed(3)
    v14 = model_v14.ChordCRNN(61, level_input=False)
    for (a, p), (b, q) in zip(v8.state_dict().items(), v14.state_dict().items()):
        assert a == b and torch.equal(p, q)
    x = torch.rand(1, 1, 144, 30)
    torch.testing.assert_close(v8.eval()(x), v14.eval()(x))


@pytest.mark.parametrize("level_input", [False, True])
def test_streaming_equals_the_whole_song(level_input):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = model_v14.ChordCRNN(61, level_input=level_input).eval()
    n, a = 25, 2
    song = LevelSong("x", rng.random((168, n)).astype(np.float32), np.zeros(n, dtype=int),
                     rng.uniform(-1, 0, n).astype(np.float32))
    offline = train_v14.song_posteriors(model, song, a)
    live = model_v14.StreamingCRNN(model, a)
    crop = song.cqt[data_v5.CENTRE_ROW:data_v5.CENTRE_ROW + data_v5.PATCH_BINS]
    got = [live.step(crop[:, t], song.level[t] if level_input else None) for t in range(n)]
    got = [p for p in got if p is not None] + live.flush()
    np.testing.assert_allclose(np.stack(got, axis=1), offline, atol=1e-5)
    assert offline.shape == (61, n)


def test_the_level_changes_the_output_and_is_required():
    torch.manual_seed(0)
    model = model_v14.ChordCRNN(61, level_input=True).eval()
    x = torch.rand(1, 1, 144, 30)
    loud, quiet = model(x, torch.zeros(1, 16)), model(x, torch.full((1, 16), -1.0))
    assert not torch.allclose(loud, quiet)
    with pytest.raises(ValueError):
        model(x)


def test_excerpt_level_is_the_newest_frame_and_silence_outside():
    song = LevelSong("x", np.zeros((168, 6), dtype=np.float32), np.zeros(6, dtype=int),
                     np.array([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4], dtype=np.float32))
    np.testing.assert_allclose(train_v14.excerpt_level(song, -1, 6, 2), [-0.8, -0.7, -0.6, -0.5, -0.4, -1.0])


@pytest.mark.parametrize("level_input", [False, True])
def test_one_epoch_trains_n(level_input):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    songs = [LevelSong(f"s{i}", rng.random((168, 140)).astype(np.float32), rng.integers(-1, 61, 140),
                       rng.uniform(-1, 0, 140).astype(np.float32)) for i in range(2)]
    cfg = {"train": {"seed": 0, "lr": 1e-3, "max_epochs": 2, "patience": 5, "dropout": 0.5, "excerpt_frames": 64,
                     "batch_excerpts": 2, "input_lookahead": 2, "level_input": level_input}}
    model, history = train_v14.train(songs, {"val": songs[:1]}, cfg, 61, N)
    assert len(history) == 2 and np.isfinite(history[-1]["train_loss"])
    assert model.level_input == level_input
    assert train_v14.song_posteriors(model, songs[0], 2).shape == (61, 140)


def test_cached_levels_must_match_the_cqt(tmp_path, monkeypatch):
    monkeypatch.setattr(features_v14, "cache_dir", lambda: tmp_path)
    (tmp_path / "set").mkdir()
    np.save(tmp_path / "set" / "a.npy", np.zeros(5, dtype=np.float32))
    assert features_v14.cached_levels(tmp_path / "none.wav", "set/a", {}, 5).shape == (5,)
    with pytest.raises(ValueError):
        features_v14.cached_levels(tmp_path / "none.wav", "set/a", {}, 6)
