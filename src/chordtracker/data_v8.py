"""Component 30 (V8): the richer chord vocabularies as training targets.

    VOCABULARIES                       name -> its qualities, in class-block order
    target(label, vocab, with_n)       Harte label -> quality * 12 + root, or -1 (not trained, not scored)
    class_names(vocab, with_n)         the Harte label of every class, e.g. 'C:maj' .. 'B:min7' for A
    n_chord_classes(vocab)             12 * qualities: the chord classes, and N's index when with_n
    frame_targets(annotation, times, vocab, with_n)
    retarget(song, annotation, cfg, vocab, with_n)   the same Song with targets in vocab
    shift_targets = data_v5.shift_targets    the root moves within its quality block; pass n_chords to keep N

Vocabularies:
    A   maj, min, 7, maj7, min7            60 classes: MIREX sevenths (mir_eval.chord.sevenths)
    B   A + maj6, sus4                     84 classes
    B9  B + 9 (dominant 9th)               96 classes

Class index = quality * 12 + root, so the first 24 classes are V7.3's (C:maj .. B:maj, C:min .. B:min).

N (V14): with_n=True adds one class after the chords, 12 * n_qualities (60 for A), for the label N. It
has no root, so a pitch shift leaves it where it is: shift_targets(y, k, n_chord_classes(vocab)). X,
chords outside the vocabulary and frames outside the annotation stay -1. with_n=False (the default) is V8.

Reading a label: the bass is ignored (C:maj/3 is C:maj). Extensions (9, 11, 13 and their alterations) are
dropped, as mir_eval does before scoring, so C:9 is C:7 and C:maj(9) is C:maj; only B9 keeps the
dominant 9th, from C:9 and from C:7 with an added 9, b9 or #9. Then the notes up to the 7th decide:
    maj6 -> maj and min6 -> min in A (B keeps maj6 as its own class, min6 -> min);
    sus4 and 7sus4 (sus4(b7)) -> sus4 in B, -1 in A;
    dim, hdim7, dim7, aug, minmaj7, sus2, power chords, single notes, N and X -> -1 (N -> N with with_n).
"""
import mir_eval
import numpy as np

from chordtracker.acoustic_hmm import frame_times
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5 import Song, shift_targets  # noqa: F401  (shift_targets is part of this module's API)

VOCABULARIES = {
    "A": ("maj", "min", "7", "maj7", "min7"),
    "B": ("maj", "min", "7", "maj7", "min7", "maj6", "sus4"),
    "B9": ("maj", "min", "7", "maj7", "min7", "maj6", "sus4", "9"),
}
NOTES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

_BITMAPS = {q: tuple(mir_eval.chord.quality_to_bitmap(q)) for q in ("maj", "min", "7", "maj7", "min7", "maj6",
                                                                     "min6", "sus4")}
_SUS4_7 = tuple(int(b) for b in np.array(_BITMAPS["sus4"]) | np.eye(12, dtype=int)[10])
_NINTHS = {"9", "b9", "#9"}


def quality(label: str, vocab: str = "A") -> str | None:
    """The vocabulary quality a Harte label trains as, or None."""
    try:
        root, bitmap, _ = mir_eval.chord.encode(label, reduce_extended_chords=False)
    except mir_eval.chord.InvalidChordException:
        return None
    if root < 0:                                   # N and X
        return None
    qualities = VOCABULARIES[vocab]
    notes = tuple(int(b) for b in bitmap)
    if notes == _BITMAPS["7"] and "9" in qualities and _is_dominant_ninth(label):
        return "9"
    for q in ("maj", "min", "7", "maj7", "min7"):
        if notes == _BITMAPS[q]:
            return q
    if notes == _BITMAPS["maj6"]:
        return "maj6" if "maj6" in qualities else "maj"
    if notes == _BITMAPS["min6"]:
        return "min"
    if notes in (_BITMAPS["sus4"], _SUS4_7) and "sus4" in qualities:
        return "sus4"
    return None


def _is_dominant_ninth(label: str) -> bool:
    """C:9, or C:7 with an added 9, b9 or #9 (C:7(#9)); C:11 and C:13 stay 7."""
    _, shorthand, extensions, _ = mir_eval.chord.split(label, reduce_extended_chords=False)
    return shorthand == "9" or (shorthand == "7" and bool(_NINTHS & {e.strip() for e in extensions}))


def target(label: str, vocab: str = "A", with_n: bool = False) -> int:
    """Harte label -> quality * 12 + root in vocab, or -1; with with_n, 'N' -> 12 * n_qualities.

    Examples (A): 'C:maj7' -> 36, 'A:min7' -> 57, 'Bb:7/3' -> 34, 'G:9' -> 31, 'D:sus4' -> -1, 'N' -> -1
    ('N' -> 60 with with_n; 'X' -> -1 either way).
    """
    if with_n and label.strip() == "N":
        return n_chord_classes(vocab)
    q = quality(label, vocab)
    if q is None:
        return -1
    root = mir_eval.chord.encode(label)[0]
    return VOCABULARIES[vocab].index(q) * 12 + int(root)


def n_chord_classes(vocab: str = "A") -> int:
    """The number of chord classes of vocab (60 for A); with N, N is the class at this index."""
    return 12 * len(VOCABULARIES[vocab])


def class_names(vocab: str = "A", with_n: bool = False) -> list[str]:
    """The Harte label of every class, in index order: 'C:maj', 'C#:maj', ..., 'B:min7' for A, then 'N' with with_n."""
    return [f"{note}:{q}" for q in VOCABULARIES[vocab] for note in NOTES] + (["N"] if with_n else [])


def frame_targets(annotation: ChordAnnotation, times: np.ndarray, vocab: str = "A", with_n: bool = False) -> np.ndarray:
    """The target of the chord sounding at each time; -1 before the first and after the last segment."""
    classes = np.array([target(label, vocab, with_n) for label in annotation.labels])
    segment = np.searchsorted(annotation.intervals[:, 0], times, side="right") - 1
    inside = (segment >= 0) & (times < annotation.intervals[np.clip(segment, 0, None), 1])
    return np.where(inside, classes[np.clip(segment, 0, None)], -1)


def retarget(song: Song, annotation: ChordAnnotation, cfg: dict, vocab: str = "A", with_n: bool = False) -> Song:
    """The same song and CQT, with every frame's target in vocab (plus N with with_n)."""
    times = frame_times(song.cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    return Song(song.key, song.cqt, frame_targets(annotation, times, vocab, with_n))
