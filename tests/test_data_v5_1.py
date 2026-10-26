import numpy as np
import pytest

from chordtracker import data_v5
from chordtracker import data_v5_1 as d
from chordtracker.data.data import ChordAnnotation


@pytest.mark.parametrize("label, expected", [
    ("C:maj", 0), ("C:maj7", 0), ("A:min7", 21), ("Bb:7/3", 10),
    ("N", 24), ("X", -1), ("D:sus4", -1), ("E:5", -1), ("nonsense", -1),
])
def test_target_adds_n_and_keeps_everything_else(label, expected):
    assert d.target(label) == expected


@pytest.mark.parametrize("label", ["C:maj", "F#:min7", "Bb:7/3", "D:sus2", "X", "C:hdim7"])
def test_chord_targets_are_the_same_as_v5(label):
    assert d.target(label) == data_v5.target(label)


def test_frame_targets_label_n_inside_the_span_only():
    ann = ChordAnnotation(np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]), ["N", "G:maj", "N"])
    y = d.frame_targets(ann, np.array([0.5, 1.5, 2.5, 3.5]))
    assert list(y) == [24, 7, 24, -1]                    # 3.5 s is past the last segment: no target


def test_shift_targets_keeps_n():
    y = np.array([0, 21, 24, -1])
    assert list(d.shift_targets(y, 2)) == [2, 23, 24, -1]
    assert list(d.shift_targets(y, -5)) == [7, 16, 24, -1]


def test_v5_still_masks_n():
    assert data_v5.target("N") == -1
