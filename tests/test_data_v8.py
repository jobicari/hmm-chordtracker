"""Behaviour of the V8 targets: every mapping rule of each vocabulary, the class layout (the first 24 are
V7.3's), class names that read back as their own class, and shifts that stay in their quality block."""
import numpy as np
import pytest

from chordtracker import data_v5, data_v8
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5 import Song

A, B, B9 = "A", "B", "B9"
CASES = [
    # label,            A,   B,   B9
    ("C:maj",           0,   0,   0),
    ("C",               0,   0,   0),
    ("A:min",          21,  21,  21),
    ("G:7",            31,  31,  31),
    ("D:maj7",         38,  38,  38),
    ("E:min7",         52,  52,  52),
    ("Bb:7/3",         34,  34,  34),          # bass ignored
    ("C:maj/5",         0,   0,   0),
    ("G:9",            31,  31, 91),           # dominant 9th: 7 except in B9
    ("G:7(#9)",        31,  31, 91),
    ("G:7(b9)",        31,  31, 91),
    ("G:13",           31,  31,  31),          # 11ths and 13ths fold to 7 everywhere
    ("G:11",           31,  31,  31),
    ("C:maj(9)",        0,   0,   0),          # added 9th folds to the triad
    ("C:maj9",         36,  36,  36),
    ("A:min9",         57,  57,  57),
    ("A:min(9)",       21,  21,  21),
    ("F:maj6",          5,  65,  65),          # maj6: maj in A, its own class in B
    ("F:maj6(9)",       5,  65,  65),
    ("A:min6",         21,  21,  21),
    ("D:sus4",         -1,  74,  74),
    ("D:sus4(b7)",     -1,  74,  74),          # 7sus4 -> sus4
    ("D:sus2",         -1,  -1,  -1),
    ("B:dim",          -1,  -1,  -1),
    ("B:hdim7",        -1,  -1,  -1),
    ("B:dim7",         -1,  -1,  -1),
    ("C:aug",          -1,  -1,  -1),
    ("C:minmaj7",      -1,  -1,  -1),
    ("C:maj(*3)",      -1,  -1,  -1),          # power chord
    ("C:5",            -1,  -1,  -1),
    ("C:1",            -1,  -1,  -1),
    ("N",              -1,  -1,  -1),
    ("X",              -1,  -1,  -1),
    ("not a chord",    -1,  -1,  -1),
]


@pytest.mark.parametrize("label,a,b,b9", CASES)
def test_every_mapping_rule(label, a, b, b9):
    assert (data_v8.target(label, A), data_v8.target(label, B), data_v8.target(label, B9)) == (a, b, b9)


@pytest.mark.parametrize("label", ["C:maj", "F#:min", "Bb:maj7", "E:min7", "A:7/b7", "D:maj6", "G:min6", "N"])
def test_the_first_24_classes_are_v7_3s(label):
    """A chord V7.3 trains as maj or min keeps its index when A keeps it in maj or min."""
    q = data_v8.quality(label, A)
    if q in ("maj", "min"):
        assert data_v8.target(label, A) == data_v5.target(label)


@pytest.mark.parametrize("vocab,n", [(A, 60), (B, 84), (B9, 96)])
def test_class_names_read_back_as_their_class(vocab, n):
    names = data_v8.class_names(vocab)
    assert len(names) == n == 12 * len(data_v8.VOCABULARIES[vocab])
    assert [data_v8.target(name, vocab) for name in names] == list(range(n))


@pytest.mark.parametrize("vocab", [A, B, B9])
@pytest.mark.parametrize("k", [-5, -1, 1, 6])
def test_a_shift_moves_the_root_within_its_quality_block(vocab, k):
    """B:maj7 + 1 is C:maj7, not C of the next block; the shifted target is the target of the shifted chord."""
    names = data_v8.class_names(vocab)
    y = np.array(list(range(len(names))) + [-1])
    shifted = data_v8.shift_targets(y, k)
    assert shifted[-1] == -1
    assert np.array_equal(shifted[:-1] // 12, y[:-1] // 12)
    for i, name in enumerate(names):
        root, q = name.split(":")
        moved = data_v8.NOTES[(data_v8.NOTES.index(root) + k) % 12]
        assert shifted[i] == data_v8.target(f"{moved}:{q}", vocab)


def test_frame_targets_and_retarget():
    ann = ChordAnnotation(np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]), ["C:maj7", "D:sus4", "G:9"])
    times = np.array([-0.1, 0.5, 1.5, 2.5, 3.5])
    assert data_v8.frame_targets(ann, times, A).tolist() == [-1, 36, -1, 31, -1]
    assert data_v8.frame_targets(ann, times, B9).tolist() == [-1, 36, 74, 91, -1]
    cfg = {"audio": {"sr": 22050}, "cqt": {"hop_length": 2048}}
    song = Song("toy", np.zeros((168, 30), dtype=np.float32), np.zeros(30, dtype=int))
    out = data_v8.retarget(song, ann, cfg, A)
    assert out.cqt is song.cqt and out.key == "toy"
    assert out.targets[0] == 36 and out.targets[29] == 31      # frame 29 is at 2.69 s: G:9 is G:7 in A
