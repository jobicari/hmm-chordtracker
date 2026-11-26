"""Behaviour of the bundle for a port of the live chain: arrays read back as written, the network's step and
its replay give the streaming CRNN's posteriors, system.json holds the demo's points and the decoder's
matrices, a captured recording keeps every stage of the live chain, and a bundle checks itself."""
import json

import librosa
import numpy as np
import pytest
import torch

from chordtracker import bundle, demo, model_v14, transitions
from chordtracker.play_along import INPUT_LOOKAHEAD, SILENCE_DBFS, Analysis
from chordtracker.streaming_v7_2 import LookaheadStreamingCRNN

SR, HOP, N = 22050, 2048, 97
CFG = {"version": "v15",
       "audio": {"sr": SR, "mono": True, "harmonic_margin": 4.0, "harmonic_median_frames": 9},
       "cqt": {"hop_length": HOP, "fmin": "F#1", "bins_per_octave": 24, "n_bins": 168, "tuning_seconds": 10.0},
       "compression": {"enabled": True, "level_percentile": 99, "gamma": 10}}
POINTS = demo.MODELS["v15"]


@pytest.fixture(scope="module")
def chain():
    """(model, prior, Q, labels): a seed-0 97-class network and a random decoder, V15's shapes."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    Q = rng.random((N, N))
    np.fill_diagonal(Q, 0)
    Q /= Q.sum(axis=1, keepdims=True)
    prior = rng.random(N)
    labels = [f"{r}:{q}" for q in ("maj", "min", "7", "maj7", "min7", "dim", "aug", "hdim7")
              for r in ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")] + ["N"]
    return model_v14.ChordCRNN(N).eval(), prior / prior.sum(), Q, labels


def sliding(seconds, cents=0.0, seed=0):
    """A tone moving from A3 to E4 halfway, `cents` off A440, with a little noise."""
    t = np.arange(int(seconds * SR)) / SR
    hz = np.where(t < seconds / 2, 220.0, 329.63) * 2 ** (cents / 1200)
    noise = 0.005 * np.random.default_rng(seed).standard_normal(len(t))
    return (0.2 * np.sin(2 * np.pi * hz * t) + noise).astype(np.float32)


@pytest.mark.parametrize("dtype,value,raw", [("f32", 1.0, b"\x00\x00\x80\x3f"), ("i32", 2, b"\x02\x00\x00\x00"),
                                              ("f64", -2.0, b"\x00" * 7 + b"\xc0")])
def test_arrays_are_raw_little_endian_and_read_back(tmp_path, dtype, value, raw):
    d = bundle.write_array(tmp_path, "sub/x", np.full((2, 3), value), dtype)
    assert d == {"file": f"sub/x.{dtype}", "dtype": dtype, "shape": [2, 3]}
    assert (tmp_path / d["file"]).read_bytes()[:len(raw)] == raw
    assert np.array_equal(bundle.read_array(tmp_path, d), np.full((2, 3), value))


def test_complex_kernels_read_back(tmp_path):
    k = np.array([1 + 2j, -0.5j])
    assert np.array_equal(bundle.read_array(tmp_path, bundle.write_array(tmp_path, "k", k, "c64")), k)


def test_the_replayed_step_gives_the_streaming_posteriors(chain):
    """FrameStep fed column by column (replay, flush included) = LookaheadStreamingCRNN, frame for frame."""
    model = chain[0]
    columns = np.random.default_rng(1).random((40, 168)).astype(np.float32)
    s = LookaheadStreamingCRNN(model, INPUT_LOOKAHEAD)
    want = [p for p in (s.step(c[12:156]) for c in columns) if p is not None] + s.flush()
    step = bundle.FrameStep(model)

    def predict(window, hidden):
        with torch.no_grad():
            post, hidden = step(torch.from_numpy(window), torch.from_numpy(hidden))
        return post.numpy(), hidden.numpy()
    network = {"rows": [12, 156], "windowFrames": 15, "inputLookaheadFrames": INPUT_LOOKAHEAD,
               "inputs": {"hidden": [1, 1, 64]}}
    got = bundle.replay(predict, columns, network)
    assert got.shape == (40, N)
    np.testing.assert_allclose(got, np.stack(want), atol=1e-6)


def test_the_description_has_the_demos_points_and_their_matrices(chain):
    _, prior, Q, labels = chain
    d, arrays = bundle.describe("v15", CFG, "results/v15/ft_seed0", prior, Q, labels, POINTS, {"commit": "abc"})
    assert [p["id"] for p in d["decoder"]["points"]] == ["0.28 s", "0.46 s", "0.65 s", "1.02 s"]
    assert d["decoder"]["defaultPoint"] == "0.65 s" and d["decoder"]["framewisePoint"] == "0.28 s"
    info = {p["id"]: p for p in demo.point_info(SR, HOP, POINTS)}
    for p in d["decoder"]["points"]:
        v = POINTS[p["id"]]
        assert p["playbackDelaySeconds"] == info[p["id"]]["playbackDelaySeconds"]
        assert p["lookaheadFrames"] == INPUT_LOOKAHEAD + 1 + v["lag"]
        if v["p_self"] is None:
            assert p["transitions"] is None
        else:
            A = arrays[p["transitions"]["file"].removesuffix(".f64")]
            assert np.array_equal(A, transitions.transition(Q, v["p_self"]))
            np.testing.assert_allclose(A.sum(axis=1), 1.0)
    assert d["labels"][-1] == {"harte": "N", "name": "N.C."} and d["labels"][-2]["name"] == "Bm7b5"
    assert d["network"]["rows"] == [12, 156] and d["network"]["classes"] == N
    assert d["source"] == {"commit": "abc", "config": "configs/v15.yaml", "run": "results/v15/ft_seed0"}
    assert np.array_equal(arrays["decoder/prior"], prior)


def test_the_tuning_pitches_give_librosas_estimate():
    y = sliding(3.0, cents=-30)
    f = bundle.tuning_pitches(y, SR)
    assert len(f) > 0
    assert librosa.pitch_tuning(f, resolution=0.01, bins_per_octave=12) == librosa.estimate_tuning(y=y, sr=SR, bins_per_octave=12)


@pytest.fixture(scope="module")
def captured(chain):
    """A 12 s tone, 30 cents flat after 1 s of silence, through the capture (the tuning step included)."""
    model, prior, Q, _ = chain
    y = np.concatenate([np.zeros(SR, dtype=np.float32), sliding(11.5, cents=-30)])
    return y, bundle.capture(y, CFG, model, prior, Q, POINTS, N - 1, SILENCE_DBFS, 1000)


def test_a_capture_has_the_live_chains_labels(chain, captured):
    model, prior, Q, _ = chain
    y, (scalars, arrays) = captured
    a = Analysis(CFG, model, prior, Q, POINTS, None, N - 1, SILENCE_DBFS)
    for i in range(0, len(y), 2048):
        a.step(y[i:i + 2048])
    a.finish()
    T = scalars["frames"]
    assert T == 1 + len(y) // HOP
    for point, v in POINTS.items():
        assert arrays[f"labels/lag{v['lag']}"][0].tolist() == a.labels[point]
    assert arrays["posteriors"][0].shape == (T, N) and arrays["logcqt"][0].shape == (T, 168)
    assert np.array_equal(arrays["posteriors"][0].argmax(axis=1), arrays["labels/lag0"][0])


def test_a_capture_keeps_every_stage(captured):
    y, (scalars, arrays) = captured
    T = scalars["frames"]
    assert arrays["input"][0] is y and len(arrays["harmonic"][0]) == len(y) and len(arrays["levels"][0]) == T
    assert arrays["cqt"][0].shape == (T, 168)
    assert scalars["streamedVsRecomputedLogcqt"] < 1e-4
    t = scalars["tuning"]
    start = (SR // HOP) * HOP                                         # the first frame with tone in it
    assert t["startSample"] == start
    assert -0.35 < t["semitones"] < -0.25        # 30 cents flat: past the 25 cents a 24-bin estimate wraps at
    assert t["bins"] == pytest.approx(2 * t["semitones"])
    assert t["firstTunedColumn"] == int(np.ceil((start + 10 * SR) / HOP)) - 1
    assert len(arrays["tuningHarmonic"][0]) == 10 * SR
    assert librosa.pitch_tuning(arrays["tuningPitches"][0], resolution=0.01, bins_per_octave=12) == t["semitones"]
    bins = scalars["cqtBins"]
    assert bins["a440"]["hz"][0] == pytest.approx(librosa.note_to_hz("F#1"))
    assert bins["tuned"]["hz"][0] == pytest.approx(librosa.note_to_hz("F#1") * 2 ** (t["semitones"] / 12))
    assert len(arrays["kernelA440Bin0"][0]) == bins["a440"]["taps"][0]


def test_a_short_recording_has_no_tuning(chain):
    model, prior, Q, _ = chain
    scalars, arrays = bundle.capture(sliding(2.0), CFG, model, prior, Q, POINTS, N - 1, SILENCE_DBFS, 2048)
    assert scalars["tuning"]["bins"] is None and scalars["cqtBins"]["tuned"] is None
    assert "tuningHarmonic" not in arrays


def test_a_bundle_checks_itself(tmp_path, chain, captured):
    """describe + write_case, then check with the step in torch: the golden posteriors and every label back."""
    model, prior, Q, labels = chain
    d, arrays = bundle.describe("v15", CFG, "run", prior, Q, labels, POINTS, {})
    for name, array in arrays.items():
        bundle.write_array(tmp_path, name, array, "f64")
    (tmp_path / "system.json").write_text(json.dumps(d))
    _, (scalars, case) = captured
    bundle.write_case(tmp_path, "tone", scalars, case, {})
    step = bundle.FrameStep(model)

    def predict(window, hidden):
        with torch.no_grad():
            post, hidden = step(torch.from_numpy(window), torch.from_numpy(hidden))
        return post.numpy(), hidden.numpy()
    result = bundle.check(tmp_path, predict)["tone"]
    assert result["posteriorsMaxAbsDiff"] < 1e-5
    assert result["labelsAgree"] == {p: 1.0 for p in POINTS}
