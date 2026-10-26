"""V5.1: V5's frame targets with a 25th class, N (no chord). Everything else is data_v5.

    target(label)                Harte label -> 0..23 as in V5, 24 for N, or -1
    frame_targets(annotation, t) the target at each time; -1 outside the annotated span
    load(take, cfg)              -> data_v5.Song with 25-class targets
    shift_targets(y, k)          the targets k semitones higher: N stays N, -1 stays -1

Why a separate module: V5 is closed, so data_v5 keeps its 24 classes and V5 can still be rerun as
tagged. Takes, patches and the log-CQT cache are reused from data_v5 unchanged.

What gets N: only the label "N" (silence, count-ins, fade-outs). X (a chord the annotator could not
name) and chords without a third (sus2, sus4, "C:5") stay -1, as in V5: they are sounding music, so
calling them N would teach the network that some chords are silence. Frames outside the annotated
span stay -1 too, since nobody labelled them.

N has no root, so pitch shifting leaves it at 24. RWC-P train has about 379 s of N (2.6% of its
annotated time); GuitarSet has none, so its frames only ever show the network chords.
"""
import numpy as np

from chordtracker import data_v5
from chordtracker.acoustic_hmm import frame_times
from chordtracker.data.data import ChordAnnotation
from chordtracker.features_v5 import cached_log_cqt

N_CLASSES = 25
N_CLASS = 24                     # after C:maj..B:maj (0..11) and C:min..B:min (12..23)


def target(label: str) -> int:
    """Harte label -> class: 0..23 as data_v5.target, 24 for 'N', -1 for X and chords with no third.

    Examples: 'C:maj7' -> 0, 'A:min7' -> 21, 'N' -> 24, 'X' -> -1, 'D:sus4' -> -1.
    """
    if label.strip() == "N":
        return N_CLASS
    return data_v5.target(label)


def frame_targets(annotation: ChordAnnotation, times: np.ndarray) -> np.ndarray:
    """The target of the segment sounding at each time; -1 before the first and after the last segment."""
    classes = np.array([target(label) for label in annotation.labels])
    segment = np.searchsorted(annotation.intervals[:, 0], times, side="right") - 1
    inside = (segment >= 0) & (times < annotation.intervals[np.clip(segment, 0, None), 1])
    return np.where(inside, classes[np.clip(segment, 0, None)], -1)


def load(take: data_v5.Take, cfg: dict) -> data_v5.Song:
    """Cached log-CQT of the take (the V5 cache, unchanged), plus a 25-class target per frame."""
    cqt = cached_log_cqt(take.audio, take.key, cfg)
    times = frame_times(cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    return data_v5.Song(take.key, cqt, frame_targets(take.annotation, times))


def shift_targets(y: np.ndarray, k: int) -> np.ndarray:
    """Targets of the same frames played k semitones higher: chords move as in V5, N and -1 stay.

    Example: k = 2 -> C:maj (0) becomes D:maj (2), A:min (21) becomes B:min (23), N (24) stays 24.
    """
    return np.where(y == N_CLASS, N_CLASS, data_v5.shift_targets(y, k))
