"""The demo's JSON: segments cut as the WCSR cuts them, and scores that match evaluate."""
import numpy as np
import pytest

from chordtracker import demo, evaluate
from chordtracker.data.data import ChordAnnotation

LABELS = ["C:maj", "G:maj", "A:min"]
HOP, SR = 2048, 22050
STEP = HOP / SR


def test_path_segments_boundaries_halfway_between_frames():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    segs = demo.path_segments(np.array([0, 0, 1, 1]), LABELS, times, 0.4)
    assert segs == [{"start": 0.0, "end": 0.15, "label": "C:maj", "name": "C"},
                    {"start": 0.15, "end": 0.4, "label": "G:maj", "name": "G"}]


def test_reference_segments_keep_the_label_and_show_the_musician_name():
    ref = ChordAnnotation(np.array([[0.0, 1.0], [1.0, 2.5]]), ["E:min7", "N"])
    assert demo.segments(ref) == [{"start": 0.0, "end": 1.0, "label": "E:min7", "name": "Em7"},
                                  {"start": 1.0, "end": 2.5, "label": "N", "name": "N.C."}]


def make_sample(framewise, hmm, reference):
    return demo.sample("x", "X", "x.wav", len(hmm) * STEP, {"framewise": np.array(framewise), "hmm": np.array(hmm)},
                       LABELS, reference, "test", {}, HOP, SR)


def test_sample_scores_match_evaluate():
    duration = 8 * STEP
    reference = ChordAnnotation(np.array([[0.0, 3.5 * STEP], [3.5 * STEP, duration]]), ["C:maj", "A:min"])  # change halfway
    framewise = [0, 1, 0, 0, 2, 0, 2, 2]
    s = make_sample(framewise, [0, 0, 0, 0, 2, 2, 2, 2], reference)
    times = np.arange(8) * STEP
    expected = evaluate.score(reference, evaluate.path_to_annotation(np.array(framewise), LABELS, times, duration))
    assert s["systems"]["framewise"]["wcsr"] == round(100 * expected.matched_s / expected.scored_s, 1)
    assert s["systems"]["hmm"]["wcsr"] == 100.0
    assert len(s["systems"]["framewise"]["segments"]) == 6
    assert len(s["systems"]["hmm"]["segments"]) == 2


def test_sample_records_each_systems_lookahead():
    reference = ChordAnnotation(np.array([[0.0, 4 * STEP]]), ["C:maj"])
    s = make_sample([0] * 4, [0] * 4, reference)
    assert s["systems"]["framewise"]["lookahead_s"] == round(3 * STEP, 3)      # 0.28 s: input 2 + 1, no lag
    assert s["systems"]["hmm"]["lookahead_s"] == round(7 * STEP, 3)            # 0.65 s: lag 4
    assert s["systems"]["hmm"]["lag_frames"] == 4


def test_paths_of_different_lengths_are_refused():
    reference = ChordAnnotation(np.array([[0.0, 1.0]]), ["C:maj"])
    with pytest.raises(ValueError):
        demo.sample("x", "X", "x.wav", 1.0, {"framewise": np.zeros(3, int), "hmm": np.zeros(4, int)},
                    LABELS, reference, "test", {}, HOP, SR)
