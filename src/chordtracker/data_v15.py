"""Component 40 (V15): vocabulary A+, V14's 61 classes plus diminished, augmented and half-diminished chords.

    Vocab(new=("dim", "aug", "hdim7"), merge=MERGE)   the vocabulary: which new qualities, which labels fold into them
    vocab.quality(label)              the quality a Harte label trains as ('maj' .. 'min7', a new one), or None
    vocab.target(label)               Harte label -> class in model order, or -1
    vocab.class_names()               the Harte label of every class, model order
    vocab.frame_targets(annotation, times) / vocab.retarget(song, annotation, cfg)
    vocab.shift_targets(y, k)         the targets of the same frames k semitones higher
    vocab.decode_order                model class of each decode index: the chords, then N last
    vocab.to_decode(y) / vocab.qualities / vocab.n_chords / vocab.n_classes

Model order (what the network outputs and trains on). V14's 61 classes keep their indices; the new
qualities follow N, one block of 12 roots each:

    0 .. 59     vocabulary A: maj, min, 7, maj7, min7 (data_v8, quality * 12 + root)
    60          N
    61 + 12 j + root    new quality j (61 .. 72 C:dim .. B:dim, 73 .. 84 aug, 85 .. 96 hdim7 by default)

Decode order (what the decoder, the prior, the matrices and the scores use): the same chords with N
moved last, so V14's code that wants N after the chords works as it is (transitions_v8 add_n,
training_prior with n_chords, frame_score columns): decode index d is model class decode_order[d], and
for vocabulary A (no new quality) the two orders are the same. Quality blocks in decode order are
vocab.qualities = ('maj', 'min', '7', 'maj7', 'min7', *new).

Reading a label is data_v8's (bass ignored, extensions above the octave dropped), then the notes up
to the 7th:
    dim     C Eb Gb       aug     C E G#       dim7    C Eb Gb A       hdim7   C Eb Gb Bb
and three spellings mir_eval reads as other notes, folded by MERGE (the default):
    min7(b5)  C Eb Gb G Bb   mir_eval's 'C:min7(b5)' adds Gb and keeps G: a half-diminished seventh
    aug(b7)   C E G# Bb      an augmented seventh ('C:aug(b7)'), trained as aug
    dim7                     folded into dim: 136 s to train on, 1 s in validation (results/v15/census.md)
A quality in new keeps its own class even if MERGE names it (new = dim, aug, dim7, hdim7 gives 109
classes); a quality neither in new nor merged into one stays -1, as in V14.
"""
from dataclasses import dataclass, field

import mir_eval
import numpy as np

from chordtracker import data_v8
from chordtracker.acoustic_hmm import frame_times
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5 import Song

BASE = data_v8.VOCABULARIES["A"]
N_BASE = data_v8.n_chord_classes("A")            # 60 chord classes of A, and N's model index
NEW_BITMAPS = {"dim": (0, 3, 6), "aug": (0, 4, 8), "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10),
               "min7(b5)": (0, 3, 6, 7, 10), "aug(b7)": (0, 4, 8, 10)}
MERGE = {"dim7": "dim", "min7(b5)": "hdim7", "aug(b7)": "aug"}


@dataclass(frozen=True)
class Vocab:
    """Vocabulary A + N + the new qualities, in model order (see the module docstring)."""
    new: tuple[str, ...] = ("dim", "aug", "hdim7")
    merge: dict = field(default_factory=lambda: dict(MERGE))

    def __post_init__(self):
        for q in self.new:
            if q not in NEW_BITMAPS:
                raise ValueError(f"unknown new quality {q!r}")
        for src in self.merge:
            if src not in NEW_BITMAPS:
                raise ValueError(f"unknown merged spelling {src!r}")

    @property
    def qualities(self) -> tuple[str, ...]:
        """Quality of each 12-class block in decode order."""
        return BASE + tuple(self.new)

    @property
    def n_chords(self) -> int:
        return 12 * len(self.qualities)

    @property
    def n_classes(self) -> int:
        return self.n_chords + 1

    @property
    def n_index(self) -> int:
        """N's model index (60); in decode order it is n_chords, the last class."""
        return N_BASE

    @property
    def decode_order(self) -> np.ndarray:
        """(n_classes,) model class of every decode index: 0 .. 59, 61 .. , then 60 (N)."""
        return np.concatenate([np.arange(N_BASE), np.arange(N_BASE + 1, self.n_classes), [N_BASE]])

    def quality(self, label: str) -> str | None:
        """'maj' .. 'min7' as data_v8 reads vocabulary A, else a new quality (after MERGE), else None."""
        q = data_v8.quality(label, "A")
        if q is not None:
            return q
        try:
            root, bitmap, _ = mir_eval.chord.encode(label, reduce_extended_chords=False)
        except mir_eval.chord.InvalidChordException:
            return None
        if root < 0:
            return None
        notes = tuple(int(i) for i in np.nonzero(bitmap)[0])
        for name, want in NEW_BITMAPS.items():
            if notes == want:
                name = name if name in self.new else self.merge.get(name)
                return name if name in self.new else None
        return None

    def target(self, label: str) -> int:
        """Harte label -> model class, or -1. Examples (default): 'C:maj7' -> 36, 'N' -> 60, 'C:dim' -> 61,
        'D:dim7' -> 63, 'C#:aug' -> 74, 'B:hdim7' -> 96, 'C:min7(b5)' -> 85, 'C:sus4' -> -1, 'X' -> -1."""
        if label.strip() == "N":
            return N_BASE
        q = self.quality(label)
        if q is None:
            return -1
        root = int(mir_eval.chord.encode(label)[0])
        if q in BASE:
            return BASE.index(q) * 12 + root
        return N_BASE + 1 + 12 * self.new.index(q) + root

    def class_names(self) -> list[str]:
        """Harte label of every model class: 'C:maj' .. 'B:min7', 'N', 'C:dim' .. 'B:hdim7'."""
        return (data_v8.class_names("A", with_n=True)
                + [f"{note}:{q}" for q in self.new for note in data_v8.NOTES])

    def decode_names(self) -> list[str]:
        """Harte label of every decode index: the chords, then 'N'."""
        names = self.class_names()
        return [names[c] for c in self.decode_order]

    def frame_targets(self, annotation: ChordAnnotation, times: np.ndarray) -> np.ndarray:
        """The model class of the chord sounding at each time; -1 outside the annotation."""
        classes = np.array([self.target(label) for label in annotation.labels])
        segment = np.searchsorted(annotation.intervals[:, 0], times, side="right") - 1
        inside = (segment >= 0) & (times < annotation.intervals[np.clip(segment, 0, None), 1])
        return np.where(inside, classes[np.clip(segment, 0, None)], -1)

    def retarget(self, song: Song, annotation: ChordAnnotation, cfg: dict) -> Song:
        times = frame_times(song.cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
        return Song(song.key, song.cqt, self.frame_targets(annotation, times))

    def shift_targets(self, y: np.ndarray, k: int) -> np.ndarray:
        """Targets k semitones higher: every root moves within its block, N (60) and -1 stay.

        Example (k = 2): 0 (C:maj) -> 2, 21 (A:min) -> 23, 60 -> 60, 61 (C:dim) -> 63, 72 (B:dim) -> 62.
        """
        y = np.asarray(y)
        block = np.where(y > N_BASE, y - N_BASE - 1, y)              # index inside the chord classes
        moved = (block % 12 + k) % 12 + 12 * (block // 12)
        moved = np.where(y > N_BASE, moved + N_BASE + 1, moved)
        return np.where((y < 0) | (y == N_BASE), y, moved)

    def to_decode(self, y: np.ndarray) -> np.ndarray:
        """Model classes -> decode indices (N -> n_chords); -1 stays."""
        inverse = np.empty(self.n_classes, dtype=int)
        inverse[self.decode_order] = np.arange(self.n_classes)
        y = np.asarray(y)
        return np.where(y < 0, -1, inverse[np.clip(y, 0, None)])

    def related(self) -> dict[str, str]:
        """The A quality each new quality's output row starts from (train_v15.extend): the closest A
        chord by its notes and its third. dim, dim7 -> min; aug -> maj; hdim7 -> min7."""
        return {"dim": "min", "dim7": "min", "aug": "maj", "hdim7": "min7"}


def vocab_from_cfg(cfg: dict) -> Vocab:
    """configs/v15.yaml: vocab_plus.new (the new qualities) and vocab_plus.merge (spelling -> quality)."""
    v = cfg.get("vocab_plus", {})
    return Vocab(tuple(v.get("new", ("dim", "aug", "hdim7"))), dict(v.get("merge", MERGE)))
