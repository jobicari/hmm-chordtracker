"""Behaviour of the sevenths metric in evaluate.score: 7ths must match, extensions fold, chords outside
MIREX sevenths are excluded."""
import numpy as np
import pytest

from chordtracker import evaluate
from chordtracker.data.data import ChordAnnotation


def ann(*labels):
    return ChordAnnotation(np.array([[2.0 * i, 2.0 * i + 2] for i in range(len(labels))]), list(labels))


@pytest.mark.parametrize("ref,est,matched", [("C:7", "C:7", 2), ("C:7", "C:maj", 0), ("C:9", "C:7", 2),
                                              ("A:min7", "A:min7", 2), ("C:maj/3", "C:maj", 2)])
def test_sevenths_needs_the_seventh(ref, est, matched):
    s = evaluate.score(ann(ref), ann(est), "sevenths")
    assert (s.scored_s, s.matched_s) == (2, matched)


def test_majmin_still_reads_the_triad():
    s = evaluate.score(ann("C:7"), ann("C:maj"), "majmin")
    assert s.matched_s == 2


@pytest.mark.parametrize("ref", ["B:hdim7", "D:sus4", "C:maj6"])
def test_chords_outside_sevenths_are_excluded(ref):
    s = evaluate.score(ann(ref), ann("C:maj"), "sevenths")
    assert s.scored_s == 0 and s.excluded_s == 2
