"""Behaviour of the listening prototype's logic: chord names, the annotation carousel in vocabulary A, the
system's labels at V11's frozen points, what the window shows when, and the delay that keeps every label on
time."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from chordtracker import model_v8, play_along as pa

SR, HOP = 22050, 2048


@pytest.mark.parametrize("label,name", [("E:min7", "Em7"), ("A", "A"), ("G/3", "G"), ("N", "N.C."),
                                        ("C:maj", "C"), ("F#:min", "F#m"), ("D:7(b9)", "D7"), ("Bb:sus4/5", "Bbsus4"),
                                        ("G:7", "G7"), ("C:maj7", "Cmaj7"), ("A:min7", "Am7")])
def test_chord_names(label, name):
    assert pa.chord_name(label) == name


def test_the_system_has_the_60_classes_of_vocabulary_a():
    assert len(pa.CLASS_LABELS) == 60
    names = [pa.chord_name(label) for label in pa.CLASS_LABELS]
    assert names[:2] == ["C", "C#"] and names[12] == "Cm" and names[24 + 7] == "G7"
    assert names[36] == "Cmaj7" and names[48 + 9] == "Am7"


@pytest.mark.parametrize("label,name", [("G:7", "G7"), ("C:maj7/3", "Cmaj7"), ("A:min7", "Am7"), ("D:9", "D7"),
                                        ("C:maj6", "C"), ("E:min6", "Em"), ("F:maj(9)", "F"), ("N", "N.C."),
                                        ("X", "N.C."), ("B:dim", "(Bdim)"), ("D:sus4", "(Dsus4)"),
                                        ("C:aug", "(Caug)"), ("B:hdim7", "(Bm7b5)")])
def test_the_reference_is_reduced_to_the_same_vocabulary(label, name):
    assert pa.reference_name(label) == name


def test_the_points_are_v11s_frozen_points():
    """Keys 1-4: network alone, then lag 2, 4 and 8 at the p_self frozen in configs/v11.yaml."""
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "v11.yaml").read_text())
    frozen = [(p["lag"], p["p_self"]) for p in cfg["frozen"]["points"]]
    assert [(v["lag"], v["p_self"]) for v in pa.OPERATING_POINTS.values()] == frozen
    assert list(pa.OPERATING_POINTS) == ["0.28 s", "0.46 s", "0.65 s", "1.02 s"]


def test_the_reference_carousel_shows_the_chord_now_and_the_next_different_one(tmp_path):
    lab = tmp_path / "song.lab"
    lab.write_text("0.0 1.0 N\n1.0 2.0 E:min\n2.0 3.0 E:min\n3.0 4.0 A:7\n4.0 5.0 A:7(b9)\n5.0 6.0 D:maj7\n")
    ref = pa.Reference.from_lab(lab)
    assert ref.at(0.5) == ("N.C.", "Em")
    assert ref.at(1.5) == ("Em", "A7")
    assert ref.at(2.5) == ("Em", "A7")                   # a repeated label is not a change
    assert ref.at(3.5) == ("A7", "Dmaj7")                # A:7(b9) is A7 in the vocabulary: no change
    assert ref.at(5.5) == ("Dmaj7", "")
    assert ref.at(-1.0) == ("", "N.C.")


def test_the_song_list_has_every_annotation_by_album(tmp_path):
    for album, song in (("02_-_With_the_Beatles", "14_-_Money"), ("01_-_Please_Please_Me", "01_-_I_Saw_Her")):
        (tmp_path / album).mkdir()
        (tmp_path / album / f"{song}.lab").write_text("0 1 N\n")
    names = [n for n, _ in pa.song_list(tmp_path)]
    assert names == ["01 - Please Please Me | 01 - I Saw Her", "02 - With the Beatles | 14 - Money"]


def test_the_onset_is_the_first_block_out_of_silence():
    onset = pa.Onset(threshold=1e-3)
    onset.update(np.zeros(2048), 0.0)
    assert onset.time is None
    onset.update(np.full(2048, 0.01), 0.093)
    onset.update(np.full(2048, 0.5), 0.186)
    assert onset.time == 0.093


def test_the_window_shows_the_chords_lead_seconds_ahead():
    labels = [0] * 10 + [31] * 10                         # C:maj for frames 0..9, G:7 from frame 10
    hop_s = HOP / SR
    at = lambda t, lead: pa.shown(labels, None, None, t, lead, 0.0, SR, HOP)
    assert at(9.5 * hop_s, 0.0)["system"] == "C" and at(9.5 * hop_s, 0.0)["system_next"] == "G7"
    assert at(9.5 * hop_s, hop_s)["system"] == "G7"       # one frame of lead: the next chord shows a frame early
    assert at(25 * hop_s, 0.0) == {**at(25 * hop_s, 0.0), "system": "", "ready": False}


def test_the_reference_starts_at_the_onset_and_can_be_nudged(tmp_path):
    lab = tmp_path / "s.lab"
    lab.write_text("0.0 1.0 C\n1.0 2.0 G\n")
    ref = pa.Reference.from_lab(lab)
    show = lambda t, nudge: pa.shown([], ref, 5.0, t, 0.0, nudge, SR, HOP)["reference"]
    assert show(5.5, 0.0) == "C" and show(6.5, 0.0) == "G"
    assert show(6.05, -0.1) == "G" and show(6.05, 0.1) == "C"     # ] (positive nudge) shows the reference later


def test_the_required_delay_covers_the_total_lookahead_and_the_lead():
    assert pa.total_lookahead("0.28 s") == 3 and pa.total_lookahead("0.65 s") == 7
    assert pa.total_lookahead("1.02 s") == 11
    for point in pa.OPERATING_POINTS:                     # the name is the total look-ahead in seconds
        assert pa.total_lookahead(point) * HOP / SR == pytest.approx(float(point.split()[0]), abs=0.005)
    assert pa.required_delay("0.28 s", 0.0, SR, HOP) == pytest.approx(4 * HOP / SR)
    assert pa.required_delay("0.65 s", 0.2, SR, HOP) == pytest.approx(8 * HOP / SR + 0.2)


def test_the_analysis_labels_every_frame_at_every_point():
    """Blocks of a short clip in, one label per frame out for each point once the song is flushed in time."""
    torch.manual_seed(0)
    model = model_v8.ChordCRNN(60, bidirectional=False)
    cfg = {"audio": {"sr": SR, "mono": True, "harmonic_margin": 4.0, "harmonic_median_frames": 9},
           "cqt": {"hop_length": HOP, "fmin": "F#1", "bins_per_octave": 24, "n_bins": 168, "tuning_seconds": 10.0},
           "compression": {"enabled": True, "level_percentile": 99, "gamma": 10}}
    Q = np.full((60, 60), 1 / 59)
    np.fill_diagonal(Q, 0)
    a = pa.Analysis(cfg, model, np.full(60, 1 / 60), Q)
    y = (0.1 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR)).astype(np.float32)
    for i in range(0, len(y), 1000):                      # any block size
        a.step(y[i:i + 1000])
    n = len(a.labels["0.28 s"])
    assert n > 20
    assert len(a.labels["0.46 s"]) == n - 2 and len(a.labels["0.65 s"]) == n - 4 and len(a.labels["1.02 s"]) == n - 8
    assert all(0 <= l < 60 for l in a.labels["0.28 s"])


def test_the_system_frames_count_from_the_onset():
    labels = [0] * 5 + [7] * 5                            # analysis started at the onset (2.0 s): C, then G
    hop_s = HOP / SR
    at = lambda t: pa.shown(labels, None, 2.0, t, 0.0, 0.0, SR, HOP)["system"]
    assert at(2.0 + 2.5 * hop_s) == "C" and at(2.0 + 7.5 * hop_s) == "G" and at(1.0) == ""


def test_finishing_a_recording_gives_the_whole_file_labels():
    """Streamed in blocks and finished, every point has exactly the labels of the whole-file computation."""
    from chordtracker import online_hmm, streaming_features, train_v7_2, transitions
    from chordtracker.data_v5 import Song

    torch.manual_seed(0)
    model = model_v8.ChordCRNN(60, bidirectional=False).eval()
    cfg = {"audio": {"sr": SR, "mono": True, "harmonic_margin": 4.0, "harmonic_median_frames": 9},
           "cqt": {"hop_length": HOP, "fmin": "F#1", "bins_per_octave": 24, "n_bins": 168, "tuning_seconds": 10.0},
           "compression": {"enabled": True, "level_percentile": 99, "gamma": 10}}
    rng = np.random.default_rng(0)
    Q = rng.random((60, 60))
    np.fill_diagonal(Q, 0)
    Q /= Q.sum(axis=1, keepdims=True)
    prior = rng.random(60)
    prior /= prior.sum()
    t = np.arange(int(SR * 4)) / SR
    y = (0.1 * np.sin(2 * np.pi * np.where(t < 2, 220, 330) * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    a = pa.Analysis(cfg, model, prior, Q)
    for i in range(0, len(y), 2048):
        a.step(y[i:i + 2048])
    a.finish()
    cqt = streaming_features.offline_log_cqt(y, cfg)
    post = train_v7_2.song_posteriors(model, Song("x", cqt, np.full(cqt.shape[1], -1)), pa.INPUT_LOOKAHEAD)
    assert a.labels["0.28 s"] == list(post.argmax(axis=0))
    for point in ("0.46 s", "0.65 s", "1.02 s"):
        v = pa.OPERATING_POINTS[point]
        whole = online_hmm.fixed_lag_viterbi(post, transitions.transition(Q, v["p_self"]), prior, [v["lag"]])[v["lag"]]
        assert a.labels[point] == list(whole)


def test_the_saved_v11_student_loads_with_its_tempered_prior(tmp_path):
    """load_system: a 60-class model in eval mode and prior ** prior_power, normalised."""
    torch.manual_seed(0)
    torch.save(model_v8.ChordCRNN(60, bidirectional=False).state_dict(), tmp_path / "model.pt")
    prior = np.linspace(1, 2, 60)
    (tmp_path / "prior.json").write_text(str((prior / prior.sum()).tolist()))
    model, decode_prior = pa.load_system({"train": {"dropout": 0.5}, "decoder": {"prior_power": 0.25}}, tmp_path)
    assert not model.training and model.classifier[1].out_features == 60
    expected = prior ** 0.25
    np.testing.assert_allclose(decode_prior, expected / expected.sum())


def test_the_demo_keeps_v7_2s_three_points():
    """The demo website decodes V7.2's 24-class network at its own points; the names keep their lags."""
    assert list(pa.V7_2_POINTS) == ["0.28 s", "0.46 s", "0.65 s"]
    assert [v["p_self"] for v in pa.V7_2_POINTS.values()] == [None, 0.9, 0.98]
    for name, v in pa.V7_2_POINTS.items():
        assert pa.OPERATING_POINTS[name]["lag"] == v["lag"]
    cfg = {"audio": {"sr": SR, "mono": True, "harmonic_margin": 4.0, "harmonic_median_frames": 9},
           "cqt": {"hop_length": HOP, "fmin": "F#1", "bins_per_octave": 24, "n_bins": 168, "tuning_seconds": 10.0},
           "compression": {"enabled": True, "level_percentile": 99, "gamma": 10}}
    Q = np.full((24, 24), 1 / 23)
    np.fill_diagonal(Q, 0)
    a = pa.Analysis(cfg, model_v8.ChordCRNN(24, bidirectional=False), np.full(24, 1 / 24), Q, pa.V7_2_POINTS)
    assert list(a.labels) == list(pa.V7_2_POINTS) and set(a.decoders) == {"0.46 s", "0.65 s"}


def test_a_late_frame_keeps_the_last_chord_on_screen():
    """While the analysis is behind (the 10 s tuning step), the window keeps the last labelled chord."""
    hop_s = HOP / SR
    labels = [0] * 5 + [31] * 5                           # C, then G7; frame 10 on not labelled yet
    at = lambda t: pa.shown(labels, None, None, t, 0.0, 0.0, SR, HOP)
    last = at(9.5 * hop_s)
    assert last["ready"] and last["system"] == "G7"
    late = pa.held(at(11.5 * hop_s), last)
    assert not late["ready"] and late["system"] == "G7" and late["frame"] == 11
    assert pa.held(at(11.5 * hop_s), None)["system"] == ""            # nothing labelled yet: nothing to keep
    assert pa.held(at(-1.0), last)["system"] == ""                    # before the song: nothing shown
    assert pa.held(at(2.5 * hop_s), last)["system"] == "C"            # a labelled frame shows its own chord


GATE_CFG = {"audio": {"sr": SR, "mono": True, "harmonic_margin": 4.0, "harmonic_median_frames": 9},
            "cqt": {"hop_length": HOP, "fmin": "F#1", "bins_per_octave": 24, "n_bins": 168, "tuning_seconds": 10.0},
            "compression": {"enabled": True, "level_percentile": 99, "gamma": 10}}


def gated_analysis(y, silence_dbfs=pa.SILENCE_DBFS, block=1000):
    """V7.2's points on a random 24-class network, fed y in blocks and finished."""
    torch.manual_seed(0)
    Q = np.full((24, 24), 1 / 23)
    np.fill_diagonal(Q, 0)
    a = pa.Analysis(GATE_CFG, model_v8.ChordCRNN(24, bidirectional=False).eval(), np.full(24, 1 / 24), Q,
                    pa.V7_2_POINTS, silence_dbfs)
    for i in range(0, len(y), block):
        a.step(y[i:i + block])
    a.finish()
    return a


def tone(seconds, amplitude=0.1, hz=220.0):
    return (amplitude * np.sin(2 * np.pi * hz * np.arange(int(seconds * SR)) / SR)).astype(np.float32)


def noise(seconds, db, seed=0):
    return (10 ** (db / 20) * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)


def test_the_silence_gate_needs_three_quiet_frames():
    assert pa.silent_frames([-80, -30, -60, -60, -60, -30]).tolist() == [True, False, False, False, True, False]
    assert pa.silent_frames([-np.inf, -np.inf]).tolist() == [True, True]             # before the song counts
    assert not pa.silent_frames([-54.9] * 10).any()


@pytest.mark.parametrize("y", [np.zeros(3 * SR, np.float32), noise(3.0, -60)], ids=["digital silence", "noise -60 dBFS"])
def test_silence_is_no_chord_on_every_frame(y):
    a = gated_analysis(y)
    for labels in a.labels.values():
        assert len(labels) == 1 + len(y) // HOP and set(labels) == {24}


def test_without_the_gate_silence_shows_chords():
    a = gated_analysis(noise(3.0, -60), silence_dbfs=None)
    assert 24 not in a.labels["0.28 s"]


def test_the_gate_only_replaces_the_silent_frames():
    """Rests of 1, 2, 3 and 6 frames in a tone: N on the gated frames only, the decoders' labels elsewhere
    (they run underneath), and the gate block by block is silent_frames of the whole file's levels."""
    from chordtracker import streaming_features
    rest = lambda frames: np.zeros(frames * HOP, np.float32)
    y = np.concatenate([tone(1.0), rest(1), tone(0.5), rest(2), tone(0.5), rest(3), tone(0.5), rest(6), tone(1.0)])
    gated, plain = gated_analysis(y, block=777), gated_analysis(y, silence_dbfs=None, block=777)
    silent = pa.silent_frames(streaming_features.frame_dbfs(y, HOP))
    assert 0 < silent.sum() < len(silent) and gated.silent == silent.tolist()
    for point in pa.V7_2_POINTS:
        expected = np.where(silent, 24, plain.labels[point])
        assert gated.labels[point] == expected.tolist()


def test_the_chord_returns_with_the_first_audible_frame():
    intro = 40
    a = gated_analysis(np.concatenate([np.zeros(intro * HOP, np.float32), tone(2.0)]))
    for labels in a.labels.values():
        assert set(labels[:intro]) == {24} and 24 not in labels[intro:]


def test_quiet_but_audible_input_is_not_gated():
    assert 24 not in gated_analysis(noise(2.0, -50)).labels["0.65 s"]


def test_no_chord_reads_n_c():
    assert pa.class_name(60) == "N.C." and pa.class_name(0) == "C"
    assert pa.system_at([0, 0, 60, 60], 1) == ("C", "N.C.") and pa.system_at([60, 7], 0) == ("N.C.", "G")


def fake_v14_run(path, n_bias=0.0):
    """A V14 run folder: a seed-0 61-class network (N's output bias raised by n_bias), a prior, the counted
    chord -> N rate and run.json."""
    import json

    from chordtracker import model_v14

    torch.manual_seed(0)
    model = model_v14.ChordCRNN(61)
    with torch.no_grad():
        model.classifier[1].bias[60] += n_bias
    torch.save(model.state_dict(), path / "model.pt")
    prior = np.linspace(1, 2, 61)
    (path / "prior.json").write_text(json.dumps((prior / prior.sum()).tolist()))
    (path / "n_counts.json").write_text(json.dumps({"chord_to_n_rate": 0.02}))
    (path / "run.json").write_text(json.dumps({"n_classes": 61, "level_input": False}))
    return path


V14_CFG = {**GATE_CFG, "train": {"dropout": 0.5}, "decoder": {"prior_power": 0.25},
           "transitions": {"source": "billboard", "counts": "results/v3/billboard_counts.json", "alpha": 1, "tau": 1}}


def test_the_systems():
    assert set(pa.SYSTEMS) == {"v11", "v14", "v15"}
    assert pa.SYSTEMS["v11"].gate and not pa.SYSTEMS["v14"].gate
    assert pa.SYSTEMS["v14"].run == "results/v14/n_seed0" and pa.SYSTEMS["v14"].config == "configs/v14.yaml"
    for name, v in pa.V14_POINTS.items():                         # the same names, so the same look-aheads
        assert pa.OPERATING_POINTS[name]["lag"] == v["lag"]
    assert [v["p_self"] for v in pa.V14_POINTS.values()] == [None, 0.5, 0.8, 0.98]
    assert pa.N_LABEL == 60 and pa.class_name(pa.N_LABEL) == "N.C."


def test_the_v14_chain_has_n_last_held_back_in_the_decoder(tmp_path):
    from chordtracker import transitions_v8
    run = fake_v14_run(tmp_path)
    model, decode_prior, Q = pa.load_chain("v14", V14_CFG, run)
    assert not model.training and model.classifier[1].out_features == 61
    prior = np.array(__import__("json").loads((run / "prior.json").read_text()))
    tempered = transitions_v8.tempered_prior(prior, 0.25)
    expected = np.concatenate([tempered[:60], [tempered[60] / pa.V14_N_WEIGHT]])
    np.testing.assert_allclose(decode_prior, expected / expected.sum())          # N's evidence x0.1
    assert Q.shape == (61, 61) and np.allclose(Q.sum(axis=1), 1)
    np.testing.assert_allclose(Q[:60, 60], 0.02 * pa.V14_RATE_FACTOR)            # chord -> N, half the counted rate
    with pytest.raises(ValueError):
        pa.load_chain("v99", V14_CFG, run)


def test_the_v11_chain_is_the_one_the_script_used(tmp_path):
    """load_chain("v11") = load_system and run_v8.decoder_parts on the training prior, as before the switch."""
    import importlib.util
    import json
    import sys
    spec = importlib.util.spec_from_file_location("run_v8", Path(__file__).parents[1] / "scripts" / "run_v8.py")
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    run_v8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_v8)
    torch.manual_seed(0)
    torch.save(model_v8.ChordCRNN(60, bidirectional=False).state_dict(), tmp_path / "model.pt")
    prior = np.linspace(1, 2, 60)
    (tmp_path / "prior.json").write_text(json.dumps((prior / prior.sum()).tolist()))
    _, decode_prior, Q = pa.load_chain("v11", V14_CFG, tmp_path)
    np.testing.assert_allclose(Q, run_v8.decoder_parts(V14_CFG, "A", prior / prior.sum()))
    np.testing.assert_allclose(decode_prior, pa.load_system(V14_CFG, tmp_path)[1])


def v14_analysis(run, y, silence_dbfs=None):
    model, decode_prior, Q = pa.load_chain("v14", V14_CFG, run)
    a = pa.Analysis(GATE_CFG, model, decode_prior, Q, pa.V14_POINTS, silence_dbfs, n_label=pa.N_LABEL)
    for i in range(0, len(y), 1000):
        a.step(y[i:i + 1000])
    a.finish()
    return a


def test_with_v14_n_comes_from_the_network_when_the_gate_is_off(tmp_path):
    """Silence, gate off: the labels are the network's own; a network leaning to N says N on every frame."""
    a = v14_analysis(fake_v14_run(tmp_path, n_bias=20.0), np.zeros(2 * SR, np.float32))
    assert a.silent == [] and all(set(labels) == {60} for labels in a.labels.values())


def test_with_v14_the_gate_gives_the_same_n_label(tmp_path):
    a = v14_analysis(fake_v14_run(tmp_path, n_bias=-20.0), np.zeros(2 * SR, np.float32), silence_dbfs=pa.SILENCE_DBFS)
    assert all(set(labels) == {60} for labels in a.labels.values())           # N from the gate, not the network
    b = v14_analysis(tmp_path, np.zeros(2 * SR, np.float32))
    assert 60 not in b.labels["0.28 s"]


def test_v14_streamed_gives_the_whole_file_labels(tmp_path):
    """Streamed in blocks and finished, every V14 point has the labels of the whole-file computation."""
    from chordtracker import online_hmm, streaming_features, train_v7_2, transitions
    from chordtracker.data_v5 import Song

    run = fake_v14_run(tmp_path)
    t = np.arange(int(SR * 4)) / SR
    y = (0.1 * np.sin(2 * np.pi * np.where(t < 2, 220, 330) * t)).astype(np.float32)
    a = v14_analysis(run, y)
    model, decode_prior, Q = pa.load_chain("v14", V14_CFG, run)
    cqt = streaming_features.offline_log_cqt(y, GATE_CFG)
    post = train_v7_2.song_posteriors(model, Song("x", cqt, np.full(cqt.shape[1], -1)), pa.INPUT_LOOKAHEAD)
    assert post.shape[0] == 61 and a.labels["0.28 s"] == list(post.argmax(axis=0))
    for point in ("0.46 s", "0.65 s", "1.02 s"):
        v = pa.V14_POINTS[point]
        whole = online_hmm.fixed_lag_viterbi(post, transitions.transition(Q, v["p_self"]), decode_prior, [v["lag"]])
        assert a.labels[point] == list(whole[v["lag"]])


def test_v15_is_the_fine_tuned_network_in_decode_order():
    """V15's outputs are put in decode order (N last), so its labels are its vocabulary's decode names."""
    from chordtracker import data_v15, model_v14
    assert pa.SYSTEMS["v15"].run == "results/v15/ft_seed0" and not pa.SYSTEMS["v15"].gate
    vocab = data_v15.Vocab()
    cfg = {"vocab_plus": {"new": list(vocab.new), "merge": dict(data_v15.MERGE)}}
    names = pa.system_labels("v15", cfg)
    assert names == vocab.decode_names() and len(names) == 97 and names[-1] == "N"
    assert pa.system_labels("v14", {}) == pa.CLASS_LABELS + ["N"]
    torch.manual_seed(0)
    model = model_v14.ChordCRNN(97).eval()
    x = torch.randn(1, 15, 64)
    before = model.classifier(x)
    pa.decode_order_head(model, vocab.decode_order)
    assert torch.equal(model.classifier(x), before[..., torch.as_tensor(vocab.decode_order)])
    assert model.classifier[1].out_features == 97                       # the layers themselves untouched


def test_the_window_names_any_vocabulary():
    names = ["C:maj", "C:dim", "B:hdim7", "N"]
    assert [pa.class_name(i, names) for i in range(4)] == ["C", "Cdim", "Bm7b5", "N.C."]
    assert pa.system_at([1, 1, 2], 0, names) == ("Cdim", "Bm7b5")
    assert pa.class_name(60) == "N.C." and pa.class_name(0) == "C"     # vocabulary A without names
