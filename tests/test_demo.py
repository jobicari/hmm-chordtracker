"""The demo's JSON: segments cut as the WCSR cuts them, framewise left as predicted, metadata from the run."""
import numpy as np
import pytest

from chordtracker import demo
from chordtracker.data.data import ChordAnnotation

LABELS = ["C:maj", "G:maj", "A:min"]
HOP, SR = 2048, 22050
STEP = HOP / SR
META = {"sampleRate": SR, "hopLength": HOP}


def make(decoded, framewise, reference=None):
    return demo.result("x", "X", "x.wav", len(decoded) * STEP, np.array(decoded), np.array(framewise), LABELS,
                       reference, "test" if reference is not None else None, {}, META)


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
    r = make([0, 0, 1, 1, 2, 2, 2, 0], [0, 1, 0, 1, 2, 0, 2, 0])
    for key in ("decoded", "framewise"):
        segs = r[key]
        assert segs[0]["start"] == 0.0 and segs[-1]["end"] == r["duration"]
        assert all(a["end"] == b["start"] for a, b in zip(segs, segs[1:]))


def test_framewise_keeps_every_single_frame_run():
    r = make([0] * 6, [0, 1, 0, 2, 2, 0])
    assert [s["chord"] for s in r["framewise"]] == ["C:maj", "G:maj", "C:maj", "A:min", "C:maj"]
    assert [s["chord"] for s in r["decoded"]] == ["C:maj"]


def test_metadata_counts_frames_and_names_the_decoder():
    m = make([0] * 5, [0] * 5)["metadata"]
    assert m["frameCount"] == 5 and m["sampleRate"] == SR and m["hopLength"] == HOP
    assert m["decoderName"] == "fixed-lag Viterbi (lag 4 frames, p_self 0.98)"
    assert m["decodedLookaheadSeconds"] == round(7 * STEP, 3)      # input 2 + 1 + lag 4
    assert m["framewiseLookaheadSeconds"] == round(3 * STEP, 3)


def test_no_reference_is_none_not_empty():
    r = make([0] * 3, [0] * 3)
    assert r["reference"] is None and r["referenceSource"] is None


def test_outputs_of_different_runs_are_refused():
    with pytest.raises(ValueError):
        make([0] * 4, [0] * 3)
