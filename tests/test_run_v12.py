"""Behaviour of run_v12: the offsets step runs V10's method on V11's pool students and writes to results/v12."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_v10  # noqa: E402
import run_v12  # noqa: E402
import v10_offsets  # noqa: E402


def test_frozen_p_self_reads_the_v11_frozen_points():
    assert run_v12.frozen_p_self(2) == 0.3
    assert run_v12.frozen_p_self(4) == 0.7
    assert run_v12.frozen_p_self(8) == 0.98


def test_offsets_step_points_at_the_v11_pool_students(monkeypatch):
    seen = {}
    monkeypatch.setattr(run_v10, "config", lambda: {"cfg": True})
    monkeypatch.setattr(run_v10, "load_songs", lambda: {"song": {}})
    monkeypatch.setattr(v10_offsets, "run", lambda cfg, songs, **kw: seen.update(kw))
    run_v12.main(["offsets"])
    assert {s: p.name for s, p in seen["seeds"].items()} == {0: "pool_kd1_seed0", 1: "pool_kd1_seed1"}
    assert seen["seeds"][1].parent.name == "v11"
    assert seen["share_p_self"] == 0.98 and seen["ensembles"] is False and seen["name"] == "offsets"


def test_write_goes_to_results_v12(monkeypatch, tmp_path):
    monkeypatch.setattr(run_v10.paths, "REPO_ROOT", tmp_path)
    run_v12.write("offsets", "| table |", {"a": np.float64(1.0)})
    assert (tmp_path / "results" / "v12" / "offsets" / "table.md").read_text() == "| table |\n"
    assert (tmp_path / "results" / "v12" / "offsets" / "results.json").exists()


def test_share_table_names_the_set():
    rows = {"seed 0": {"reference": [50, 30, 10, 5, 5], "network": [60, 30, 5, 3, 2], "lag 8": [62, 30, 4, 2, 2]}}
    text = v10_offsets.share_table(rows)
    assert text.startswith("| RWC-P val, share of frames %")
    assert "| reference | 50.0 | 30.0 | 10.0 | 5.0 | 5.0 |" in text and "| seed 0, lag 8 | 62.0" in text


def test_tuning_match_names_the_estimate_the_cache_agrees_with():
    import v12_tuning
    assert v12_tuning.matched(5e-5, 0.05) == "24 bins"
    assert v12_tuning.matched(0.05, 5e-5) == "12 bins"
    assert v12_tuning.matched(5e-5, 6e-5) == "both"
    assert v12_tuning.matched(0.05, 0.06) == "neither"


def test_the_beat_decoder_is_fixed_lag_viterbi_when_beats_change_nothing():
    import v12_beats
    from chordtracker import online_hmm, transitions
    rng = np.random.default_rng(0)
    post = rng.dirichlet(np.ones(6), size=40).T
    Q = rng.random((6, 6))
    np.fill_diagonal(Q, 0)
    Q /= Q.sum(axis=1, keepdims=True)
    A, prior = transitions.transition(Q, 0.9), np.full(6, 1 / 6)
    on = rng.random(40) < 0.3
    for lag in (0, 3, 8):
        expected = online_hmm.fixed_lag_viterbi(post, A, prior, [lag])[lag]
        np.testing.assert_array_equal(v12_beats.beat_fixed_lag_viterbi(post, A, A, prior, on, lag), expected)


def test_a_cheap_change_on_beats_moves_the_change_onto_the_beat():
    """Evidence that turns from chord 0 to chord 1 halfway between two beats: with changes cheap only on
    beats and dear elsewhere, the decoded change lands on a beat frame."""
    import v12_beats
    from chordtracker import transitions
    T = 20
    post = np.full((2, T), 0.5)
    post[0, :9], post[1, :9] = 0.6, 0.4
    post[0, 9:], post[1, 9:] = 0.4, 0.6
    Q = np.array([[0.0, 1.0], [1.0, 0.0]])
    on = np.zeros(T, dtype=bool)
    on[[4, 12]] = True
    path = v12_beats.beat_fixed_lag_viterbi(post, transitions.transition(Q, 0.5), transitions.transition(Q, 0.999),
                                            np.full(2, 0.5), on, 8)
    change = int(np.nonzero(np.diff(path))[0][0]) + 1
    assert on[change]


def test_beat_frames_are_marked_within_the_width():
    import v12_beats
    on = v12_beats.near(np.array([0.0, 1.0]), 20, 1, 22050, 2048)     # 1.0 s is frame 10.77 -> 11
    assert np.nonzero(on)[0].tolist() == [0, 1, 10, 11, 12]
