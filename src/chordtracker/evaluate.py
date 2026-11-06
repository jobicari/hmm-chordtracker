"""Component 5: evaluation, from a decoded state path to the numbers the project reports.

Plain functions on purpose: evaluation is the yardstick, so there is one way to do it and every
version (V0 framewise, V0 Viterbi, V1, ...) is measured with exactly the same code.

    path_to_annotation(path, labels, frame_times, duration)  state per frame -> timed chord segments
    changes_per_minute(annotation, duration)                 how often the estimate changes chord
    score(reference, estimate, metric)                       duration-weighted comparison (one song)
    pooled(scores)                                           WCSR over several songs

WCSR (weighted chord symbol recall) = matched duration / scored duration, where a stretch of time
is "matched" when the estimated chord equals the reference chord under the metric.

Label policy (V0, metric = "majmin")
------------------------------------
The V0 model can only output 25 labels: 12 major triads, 12 minor triads and N. The GuitarSet
lead sheets use richer chords, so mir_eval.chord.majmin compares them through their triad:

    reference            compared as   how it is counted
    C:maj, A:min, N      itself        scored, exact match needed
    C:7, C:maj7, C:maj6  C:maj         scored, reduced: an estimate of C:maj counts as right
    A:min7, A:min6       A:min         scored, reduced
    C:maj/3              C:maj         scored, reduced (the bass note is ignored)
    B:hdim7, D:sus4, X   (none)        not scored: left out of scored_s, reported in excluded_s

So a V0 WCSR answers "did the model find the right triad?", not "did it find the written chord?".
Every SongScore keeps both effects visible instead of dropping them silently:
    reduced_s   scored seconds whose reference chord was simplified to its triad (C:7 -> C:maj)
    excluded_s  seconds that could not be scored at all (B:hdim7)
Sevenths (V8): METRICS also holds mir_eval.chord.sevenths (maj, min, 7, maj7, min7; extensions folded,
the bass ignored, anything else excluded). reduced_s keeps the V0 meaning (outside maj/min), so under
sevenths it counts the 7ths too; report WCSR together with excluded_s.
"""
from dataclasses import dataclass

import mir_eval
import numpy as np

from chordtracker.data.data import ChordAnnotation

METRICS = {"majmin": mir_eval.chord.majmin, "sevenths": mir_eval.chord.sevenths, "root": mir_eval.chord.root}
V0_QUALITIES = {"maj", "min"}


# ====================================================================================
# ================================     SEGMENTS     ==================================
# ====================================================================================

def path_to_annotation(path: np.ndarray, labels: list[str], frame_times: np.ndarray,
                       duration: float) -> ChordAnnotation:
    """Turn a (n_frames,) state path into timed chord segments covering [0, duration].

    Runs of the same state become one segment. A boundary between two runs sits halfway between
    the last frame of the old chord and the first frame of the new one; the first segment starts
    at 0 and the last one ends at duration.

    Example with labels ["C:maj", "G:maj"], frame times [0.0, 0.1, 0.2, 0.3] and duration 0.4:
        path [0, 0, 1, 1]  ->  [0.0, 0.15) C:maj, [0.15, 0.4) G:maj
    """
    if len(path) != len(frame_times):
        raise ValueError(f"{len(path)} frames in the path for {len(frame_times)} frame times")
    first_frames = np.concatenate([[0], np.flatnonzero(np.diff(path)) + 1])   # first frame of each run
    boundaries = (frame_times[first_frames[1:] - 1] + frame_times[first_frames[1:]]) / 2
    edges = np.concatenate([[0.0], boundaries, [duration]])
    intervals = np.stack([edges[:-1], edges[1:]], axis=1)                    # (n_segments, 2)
    return ChordAnnotation(intervals, [labels[path[frame]] for frame in first_frames])


def changes_per_minute(annotation: ChordAnnotation, duration: float) -> float:
    """Chord changes per minute: (n_segments - 1) / minutes. 3 segments in 30 s -> 2 / 0.5 = 4.0."""
    return (len(annotation.labels) - 1) / (duration / 60)


# ====================================================================================
# ================================      SCORES      ==================================
# ====================================================================================

def in_v0_vocabulary(label: str) -> bool:
    """True for labels the V0 model can output as they are: N, or a plain root-position maj/min triad.

    "C:maj", "Eb:min", "C" (shorthand for C:maj), "N" -> True
    "C:7", "C:maj7", "A:min7", "C:maj/3", "B:hdim7", "X" -> False
    """
    if label == mir_eval.chord.NO_CHORD:
        return True
    root, quality, extensions, bass = mir_eval.chord.split(label)
    return root != "X" and quality in V0_QUALITIES and not extensions and bass == "1"


@dataclass(frozen=True)
class SongScore:
    """Durations in seconds for one song; durations, not ratios, so songs can be pooled.

    total_s     length of the reference (e.g. 22.32 s)
    scored_s    part of total_s whose reference chord the metric can judge
    matched_s   part of scored_s where the estimate is right
    reduced_s   part of scored_s whose reference chord is outside the V0 vocabulary and was judged
                through its triad (C:7 -> C:maj); see the label policy at the top of the module
    excluded_s  total_s - scored_s (reference chords the metric cannot judge, e.g. B:hdim7 for majmin)

    Example, reference C:maj 0-4 s, G:7 4-6 s, B:hdim7 6-8 s:
        total_s 8, scored_s 6, reduced_s 2 (the G:7), excluded_s 2 (the B:hdim7)
    """
    total_s: float
    scored_s: float
    matched_s: float
    reduced_s: float = 0.0

    @property
    def excluded_s(self) -> float:
        return self.total_s - self.scored_s

    @property
    def wcsr(self) -> float:
        """matched_s / scored_s, e.g. 15 s right out of 20 s scored -> 0.75; nan if nothing is scored."""
        return self.matched_s / self.scored_s if self.scored_s > 0 else float("nan")


def score(reference: ChordAnnotation, estimate: ChordAnnotation, metric: str = "majmin") -> SongScore:
    """Compare an estimate with the reference over the reference's time span, weighted by duration.

    Reference chords outside the V0 vocabulary are handled as in the label policy at the top of the
    module: reduced to their triad when majmin can read them (counted in reduced_s), else excluded.

    The estimate is first clipped or padded with "N" to the reference span, then both are cut at
    every boundary of either, so each piece has one reference and one estimated label.
    Example: reference [0, 4) C:maj, [4, 8) A:min; estimate [0, 5) C:maj, [5, 8) A:min
        pieces [0, 4) C/C right, [4, 5) A:min/C:maj wrong, [5, 8) A/A right -> 7 of 8 s, WCSR 0.875
    """
    if metric not in METRICS:
        raise ValueError(f"unsupported metric {metric!r}, expected one of {sorted(METRICS)}")
    start, end = reference.intervals.min(), reference.intervals.max()
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        estimate.intervals, estimate.labels, start, end, mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    intervals, ref_labels, est_labels = mir_eval.util.merge_labeled_intervals(
        reference.intervals, reference.labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    comparison = METRICS[metric](ref_labels, est_labels)        # 1 right, 0 wrong, -1 not scored
    scored = comparison >= 0
    reduced = scored & ~np.array([in_v0_vocabulary(label) for label in ref_labels])
    return SongScore(total_s=float(durations.sum()),
                     scored_s=float(durations[scored].sum()),
                     matched_s=float((durations * comparison)[scored].sum()),
                     reduced_s=float(durations[reduced].sum()))


def pooled(scores: list[SongScore]) -> SongScore:
    """Add the durations of several songs: WCSR over the set = total matched / total scored.

    A long song weighs more than a short one, unlike the mean of per-song WCSRs.
    Example: 15 of 20 s and 5 of 10 s -> 20 / 30 = 0.667 (mean of 0.75 and 0.5 would be 0.625).
    """
    return SongScore(total_s=sum(s.total_s for s in scores),
                     scored_s=sum(s.scored_s for s in scores),
                     matched_s=sum(s.matched_s for s in scores),
                     reduced_s=sum(s.reduced_s for s in scores))
