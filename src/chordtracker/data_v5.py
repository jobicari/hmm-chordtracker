"""Component 14 (V5): what the CNN learns from, as plain functions. Songs, frame targets, patches.

A training example is one CQT frame: the 144 x 15 patch around it (input) and the chord sounding at
its centre (target). This module turns the two datasets into that, without storing any patch:

    rwc_takes(split) / guitarset_takes(split)   the songs of a split -> list[Take]
    target(label)                               Harte label -> 0..23 (C:maj..B:maj, C:min..B:min), or -1
    load(take, cfg)                             -> Song: (168, n_frames) log-CQT + one target per frame
    shift_targets(y, k, n_chords=None)          the targets of the same frames played k semitones higher
    patch(song, t, k)                           (144, 15) input for frame t, pitch-shifted by k semitones

Why patches are cut on the fly and never cached: a frame appears in 15 overlapping patches and at
12 shifts, so caching patches would store every CQT value up to 15 x 12 = 180 times. Cutting one is a
slice of the cached log-CQT (features_v5), and training draws a new random shift each time it sees a
frame, so the network meets every song in all 12 keys over the epochs.

Targets: a chord is major or minor by its third, as mir_eval reads it. C:maj7, C:7,
C:maj6 -> C:maj; C:min7, C:dim, C:hdim7 -> C:min. Chords with no third (sus2, sus4, power chords "C:5",
single notes), N and X get -1: no target, left out of the loss. The N class is a later experiment.
"""
from dataclasses import dataclass
from pathlib import Path

import mir_eval
import numpy as np

from chordtracker import paths
from chordtracker.acoustic_hmm import frame_times
from chordtracker.data.data import ChordAnnotation, load_manifest, load_split
from chordtracker.features_v5 import cached_log_cqt

RWC_DIR = paths.DATA_DIR / "rwc"
RWC_SPLIT = paths.SPLITS_DIR / "rwc_p.json"
GUITARSET_MANIFEST = paths.DATA_DIR / "processed" / "guitarset" / "manifest.csv"
GUITARSET_SPLIT = paths.SPLITS_DIR / "guitarset.json"

PATCH_BINS = 144                 # 6 octaves, C1..B6, 2 bins per semitone
PATCH_FRAMES = 15                # 1.4 s at 10.8 frames/s, centred on the frame it labels
SHIFTS = range(-5, 7)            # the 12 pitch shifts, k = -5..+6 semitones
CENTRE_ROW = 12                  # crop start for k = 0: row 12 of the 168-bin CQT is C1


@dataclass(frozen=True)
class Take:
    """One song: a cache key ("rwc/RWC_P003"), its audio file and its chord annotation."""
    key: str
    audio: Path
    annotation: ChordAnnotation


@dataclass(frozen=True)
class Song:
    """A loaded take: cqt (168, n_frames) float32 and targets (n_frames,) int, -1 = no target."""
    key: str
    cqt: np.ndarray
    targets: np.ndarray


def rwc_annotation(song_id: str) -> ChordAnnotation:
    """RWC chord csv (header t_start;t_end;chord, Harte labels) -> ChordAnnotation."""
    file = RWC_DIR / "rwc-annotations" / "01_annotations_preprocessed" / "chords" / "RWC-P" / f"{song_id}.csv"
    rows = [line.split(";") for line in file.read_text().splitlines()[1:] if line.strip()]
    return ChordAnnotation(np.array([[float(s), float(e)] for s, e, _ in rows]), [c for _, _, c in rows])


def rwc_takes(split: str) -> list[Take]:
    """The RWC-P songs of 'train' or 'val' (splits/rwc_p.json: 60/20/20, grouped by artist)."""
    if split == "test":
        raise ValueError("RWC-P test is evaluated once, by the frozen run script, never loaded here")
    return [Take(f"rwc/{s}", RWC_DIR / "RWC-P" / f"{s}.wav", rwc_annotation(s)) for s in load_split(RWC_SPLIT)[split]]


def guitarset_takes(split: str) -> list[Take]:
    """The GuitarSet comp takes of a split, from the same manifest and split file as V0-V4."""
    ids = set(load_split(GUITARSET_SPLIT)[split])
    return [Take(f"guitarset/{r.song_id}", Path(r.audio), ChordAnnotation.from_lab(r.lab))
            for r in load_manifest(GUITARSET_MANIFEST) if r.song_id in ids]


def target(label: str) -> int:
    """Harte label -> class: root for major, 12 + root for minor, -1 when there is no third or no chord.

    Examples: 'C:maj7' -> 0, 'A:min7' -> 21, 'Bb:7/3' -> 10, 'D:sus4' -> -1, 'N' -> -1.
    """
    try:
        root, bitmap, _ = mir_eval.chord.encode(label)
    except mir_eval.chord.InvalidChordException:
        return -1
    if root < 0:
        return -1
    if bitmap[4]:                    # major third, 4 semitones above the root
        return int(root)
    if bitmap[3]:                    # minor third, 3 semitones above the root
        return 12 + int(root)
    return -1


def frame_targets(annotation: ChordAnnotation, times: np.ndarray) -> np.ndarray:
    """The target of the chord sounding at each time; -1 before the first and after the last segment."""
    classes = np.array([target(label) for label in annotation.labels])
    segment = np.searchsorted(annotation.intervals[:, 0], times, side="right") - 1
    inside = (segment >= 0) & (times < annotation.intervals[np.clip(segment, 0, None), 1])
    return np.where(inside, classes[np.clip(segment, 0, None)], -1)


def load(take: Take, cfg: dict) -> Song:
    """Cached log-CQT of the take, plus the target at the centre time of every frame."""
    cqt = cached_log_cqt(take.audio, take.key, cfg)
    times = frame_times(cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    return Song(take.key, cqt, frame_targets(take.annotation, times))


def shift_targets(y: np.ndarray, k: int, n_chords: int | None = None) -> np.ndarray:
    """Targets of the same chords k semitones higher: the root moves, major stays major, -1 stays -1.

    n_chords (V14): classes from n_chords on have no root (N) and stay where they are; None shifts every class.
    Example: k = 2 -> C:maj (0) becomes D:maj (2), A:min (21) becomes B:min (23); with n_chords 60, N (60) stays 60.
    """
    shifted = np.where(y < 0, -1, (y % 12 + k) % 12 + 12 * (y // 12))
    return shifted if n_chords is None else np.where(y >= n_chords, y, shifted)


def patch(song: Song, t: int, k: int) -> np.ndarray:
    """(144, 15) input for frame t, shifted up k semitones; frames before 0 or past the end are zeros.

    Shifting up k semitones = starting the crop 2k bins lower: rows 12 - 2k .. 12 - 2k + 144 of the CQT.
    A note that sat on crop row r now sits on row r + 2k, which is where it would be if played k higher.
    """
    top = CENTRE_ROW - 2 * k
    half = PATCH_FRAMES // 2
    out = np.zeros((PATCH_BINS, PATCH_FRAMES), dtype=np.float32)
    lo, hi = max(t - half, 0), min(t + half + 1, song.cqt.shape[1])
    out[:, lo - (t - half):hi - (t - half)] = song.cqt[top:top + PATCH_BINS, lo:hi]
    return out
