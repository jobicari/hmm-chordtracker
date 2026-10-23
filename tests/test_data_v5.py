"""Behaviour of the V5 training data: targets, frame labels, pitch shift, patches."""
import numpy as np
import pytest

from chordtracker import data_v5 as d
from chordtracker.data.data import ChordAnnotation


@pytest.mark.parametrize("label, expected", [
    ("C:maj", 0), ("C:maj7", 0), ("G:7", 7), ("Bb:7/3", 10),      # major third -> major
    ("A:min", 21), ("A:min7", 21), ("B:dim", 23), ("F#:hdim7", 18),  # minor third -> minor
    ("D:sus4", -1), ("D:sus2", -1), ("E:5", -1), ("N", -1), ("X", -1), ("nonsense", -1),
])
def test_target_reads_the_third(label, expected):
    assert d.target(label) == expected


def test_frame_targets_follow_the_segments():
    ann = ChordAnnotation(np.array([[0.5, 1.0], [1.0, 2.0]]), ["G:maj", "E:min7"])
    y = d.frame_targets(ann, np.array([0.0, 0.5, 0.99, 1.0, 1.5, 2.0, 3.0]))
    assert list(y) == [-1, 7, 7, 16, 16, -1, -1]


def test_shift_targets_moves_the_root_and_keeps_the_quality():
    y = np.array([0, 11, 21, -1])
    assert list(d.shift_targets(y, 2)) == [2, 1, 23, -1]
    assert list(d.shift_targets(y, -5)) == [7, 6, 16, -1]


def toy_song(n_frames=20):
    """A CQT whose value is its row number, so a patch shows which rows it was cut from."""
    cqt = np.repeat(np.arange(168, dtype=np.float32)[:, None], n_frames, axis=1)
    return d.Song("toy", cqt, np.zeros(n_frames, dtype=int))


def test_patch_without_shift_is_c1_to_b6():
    p = d.patch(toy_song(), t=10, k=0)
    assert p.shape == (144, 15)
    assert p[0, 0] == 12 and p[-1, 0] == 155                       # rows 12..155 = C1..B6


def test_shifting_up_k_semitones_moves_content_up_2k_rows():
    song = toy_song()
    for k in d.SHIFTS:
        p = d.patch(song, t=10, k=k)
        assert p[0, 0] == 12 - 2 * k                               # a note at row r lands on row r + 2k
        assert p.shape == (144, 15)                                # always inside the 168 bins


def test_patch_is_zero_padded_at_the_song_edges():
    song = toy_song(n_frames=20)
    first = d.patch(song, t=0, k=0)
    assert (first[:, :7] == 0).all() and (first[:, 7:] > 0).all()   # 7 frames before the song start
    last = d.patch(song, t=19, k=0)
    assert (last[:, 8:] == 0).all() and (last[:, :8] > 0).all()


def test_rwc_test_songs_are_never_loaded():
    with pytest.raises(ValueError):
        d.rwc_takes("test")
