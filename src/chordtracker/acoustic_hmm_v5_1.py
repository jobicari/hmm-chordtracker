"""V5.1: the decoder's state prior with a 25th state, N. The rest of acoustic_hmm works as it is.

    LABELS                     the 25 state labels: C:maj..B:maj, C:min..B:min, N
    training_prior(targets)    (25,) P(state) the decoder divides by: chords in all 12 keys, N as is
    change_prior(counts, ...)  (25, 25) Q: V3's Billboard change prior with its N row and column kept
    n_seconds(reference, est)  where N is in the reference and in the estimate: found, missed, false

frames_to_beats and decode are acoustic_hmm's, unchanged: both take as many rows as the posteriors
have, so (25, n_frames) frame posteriors give (25, n_beats) beat posteriors and a 25-state path.

Why the prior matters for N. The decoder's evidence is P(state | beat) / P(state). N is rare in
training (about 2.4% of the frames), so dividing by its prior lifts it: where the network gives N and
a chord the same posterior, N wins if its prior is the smaller of the two. That is the point of the
division (the network learned how rare N is, the HMM should not count it twice), but it also means
any N the network half-believes on GuitarSet, which has none, gets through. Val shows which way it goes.
"""
import numpy as np

from chordtracker import acoustic_hmm, transitions
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5_1 import N_CLASS, N_CLASSES
from chordtracker.learned_emission import LABELS as CHORD_LABELS     # C:maj ... B:min, as V4 and V5

LABELS = list(CHORD_LABELS) + ["N"]


def training_prior(targets: np.ndarray) -> np.ndarray:
    """P(state): training target frequencies with every chord frame in all 12 keys and N as it is.

    Chords as in acoustic_hmm.training_prior: training sees each frame at a random shift, so every
    root is equally likely and only the major/minor balance is left. A shifted N is still N, so
    each N frame counts 12 times, once per key, and N's prior is exactly its share of the frames.
    Targets are 0..24, -1 = no target.
    Example: targets [0, 0, 12, 24] -> 2/48 for every major, 1/48 for every minor, 12/48 = 1/4 for N.
    """
    y = targets[targets >= 0]
    chords = y[y != N_CLASS]
    counts = np.zeros(N_CLASSES)
    counts[:acoustic_hmm.N_CLASSES] = acoustic_hmm.training_prior(chords) * 12 * len(chords)
    counts[N_CLASS] = 12 * np.count_nonzero(y == N_CLASS)
    return counts / counts.sum()


def change_prior(counts: np.ndarray, alpha: float = 1.0, tau: float = 1.0) -> np.ndarray:
    """(25, 25) Q = transitions.change_matrix as it is: V5 dropped its N row and column, V5.1 keeps them.

    Billboard has no N counts, so V3 fills N in by rule: a chord moves to N with the weight of its
    average chord change, and N moves to every chord equally. After the rows are normalized:
        Q[chord, N] = 1/24   (4.2% of a chord's changes go to N)
        Q[N, chord] = 1/24   (N -> any chord, uniform)
    and the 23 chord-to-chord entries of a row are V5's, scaled by 23/24, so they keep their ratios.
    With p_self 0.4: A[chord, N] = 0.6 / 24 = 2.5% per beat.
    For comparison, 95 of RWC-P train's 8,076 chord changes (1.2%) go to N: the rule gives N about
    3.5 times its share of changes. Learning these two numbers from RWC-P would be a second change, so
    V5.1 keeps V3's rule and val shows whether N flickers in and out.
    """
    return transitions.change_matrix(counts, alpha=alpha, tau=tau)


GRID_S = 0.01                    # n_seconds samples both annotations every 10 ms


def label_at(annotation: ChordAnnotation, times: np.ndarray) -> np.ndarray:
    """The label sounding at each time, "" before the first and after the last segment."""
    labels = np.array(list(annotation.labels) + [""], dtype=object)
    segment = np.searchsorted(annotation.intervals[:, 0], times, side="right") - 1
    inside = (segment >= 0) & (times < annotation.intervals[np.clip(segment, 0, None), 1])
    return np.where(inside, labels[np.clip(segment, 0, None)], "")


def n_seconds(reference: ChordAnnotation, estimate: ChordAnnotation) -> dict[str, float]:
    """Seconds of N, read on a 10 ms grid over the reference span (WCSR does not show this split).

        ref_n    seconds the reference calls N
        found    reference N, estimate N
        missed   reference N, estimate a chord (all of V5's ref_n)
        false_n  reference a chord (anything but N or X), estimate N: the cost of the N class
    Example: reference [0, 2) N, [2, 6) C:maj; estimate [0, 3) N, [3, 6) C:maj
        -> ref_n 2, found 2, missed 0, false_n 1.
    """
    times = np.arange(reference.intervals[0, 0], reference.intervals[-1, 1], GRID_S) + GRID_S / 2
    ref, est = label_at(reference, times), label_at(estimate, times)
    ref_n, est_n = ref == "N", est == "N"
    ref_chord = (ref != "N") & (ref != "X") & (ref != "")
    return {"ref_n": GRID_S * float(ref_n.sum()), "found": GRID_S * float((ref_n & est_n).sum()),
            "missed": GRID_S * float((ref_n & ~est_n).sum()), "false_n": GRID_S * float((ref_chord & est_n).sum())}
