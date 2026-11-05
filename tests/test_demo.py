"""The demo's JSON: segments cut as the WCSR cuts them, framewise left as predicted, each point's delays."""
import numpy as np
import pytest

from chordtracker import demo
from chordtracker.data.data import ChordAnnotation
from chordtracker.play_along import OPERATING_POINTS

LABELS = ["C:maj", "G:maj", "A:min"]
HOP, SR = 2048, 22050
STEP = HOP / SR
META = {"sampleRate": SR, "hopLength": HOP}


def make(framewise, points=None, reference=None):
    framewise = np.array(framewise)
    points = points or {p: framewise for p in OPERATING_POINTS}
    return demo.result("x", "X", "x.wav", len(framewise) * STEP, {k: np.array(v) for k, v in points.items()},
                       framewise, LABELS, reference, "test" if reference is not None else None, {}, META)


def point(r, pid):
    return next(p for p in r["points"] if p["id"] == pid)


def test_path_segments_boundaries_halfway_between_frames():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    segs = demo.path_segments(np.array([0, 0, 1, 1]), LABELS, times, 0.4)
    assert segs == [{"start": 0.0, "end": 0.15, "chord": "C:maj", "name": "C"},
                    {"start": 0.15, "end": 0.4, "chord": "G:maj", "name": "G"}]


def test_reference_segments_keep_the_label_and_show_the_musician_name():
    ref = ChordAnnotation(np.array([[0.0, 1.0], [1.0, 2.5]]), ["E:min7", "N"])
    assert demo.segments(ref) == [{"start": 0.0, "end": 1.0, "chord": "E:min7", "name": "Em7"},
                                  {"start": 1.0, "end": 2.5, "chord": "N", "name": "N.C."}]


def test_segments_cover_the_whole_recording_without_gaps():
    r = make([0, 1, 0, 1, 2, 0, 2, 0], {p: [0, 0, 1, 1, 2, 2, 2, 0] for p in OPERATING_POINTS})
    for segs in [r["framewise"]] + [p["segments"] for p in r["points"]]:
        assert segs[0]["start"] == 0.0 and segs[-1]["end"] == r["duration"]
        assert all(a["end"] == b["start"] for a, b in zip(segs, segs[1:]))


def test_framewise_keeps_every_single_frame_run():
    r = make([0, 1, 0, 2, 2, 0], {p: [0] * 6 for p in OPERATING_POINTS})
    assert [s["chord"] for s in r["framewise"]] == ["C:maj", "G:maj", "C:maj", "A:min", "C:maj"]
    assert [s["chord"] for s in point(r, "0.65 s")["segments"]] == ["C:maj"]


def test_every_point_has_its_lookahead_and_decision_delay():
    r = make([0] * 5)
    assert [p["id"] for p in r["points"]] == ["0.28 s", "0.46 s", "0.65 s"]
    for p, (lag, total) in zip(r["points"], [(0, 3), (2, 5), (4, 7)]):       # input 2 + 1 + lag
        assert p["lagFrames"] == lag
        assert p["lookaheadSeconds"] == round(total * STEP, 3)
        assert p["decisionDelaySeconds"] == round((total + 1) * STEP, 3)      # + the frame's own block
        assert p["playbackDelaySeconds"] == round((total + 1.5) * STEP, 3)    # + half a frame: on time
    assert point(r, "0.28 s")["pSelf"] is None and point(r, "0.65 s")["pSelf"] == 0.98
    assert r["framewiseDecisionDelaySeconds"] == point(r, "0.28 s")["decisionDelaySeconds"]
    assert r["defaultPoint"] == "0.65 s"


def test_metadata_counts_frames():
    m = make([0] * 5)["metadata"]
    assert m["frameCount"] == 5 and m["frameSeconds"] == round(STEP, 6)


def test_no_reference_is_none_not_empty():
    r = make([0] * 3)
    assert r["reference"] is None and r["referenceSource"] is None


def test_outputs_of_different_runs_are_refused():
    with pytest.raises(ValueError):
        make([0] * 4, {p: [0] * 3 for p in OPERATING_POINTS})


def test_every_point_is_required():
    with pytest.raises(ValueError):
        make([0] * 3, {"0.65 s": [0] * 3})
