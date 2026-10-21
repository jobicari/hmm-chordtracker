"""Component 10 (V3): a transition matrix learned from chord labels, as plain functions.

V0 to V2 decode with librosa.sequence.transition_loop(25, p_self): stay with p_self, otherwise
move to any of the 24 other states with the same probability (1 - p_self) / 24. V3 keeps the
self-loop p_self (chosen on val, as before) and learns where a chord goes when it changes:

    A[i, i] = p_self
    A[i, j] = (1 - p_self) * Q[i, j]        for j != i, with Q[i, i] = 0 and each row of Q summing to 1

Q is transposition-invariant: it depends only on the two qualities (maj or min) and on the root
interval, so C:maj -> G:maj and D:maj -> A:maj (both "maj -> maj, up 7 semitones") share one count.
The counts come from label files only (GuitarSet train: 420 changes from 5 lead sheets; McGill
Billboard: 58,794 changes from 730 songs). They get Laplace smoothing (alpha per cell), and a weight
tau tempers them: Q is proportional to counts ** tau. tau = 0 gives the uniform
Q of transition_loop (the V2 decoder exactly), tau = 1 the smoothed frequencies.
On val neither source beats tau = 0 (reports/v3.md): V3 keeps the V2 decoder, and this module is
kept for V4, whose calibrated posteriors can take a prior at full strength.

    chord_class(label)                       Harte label -> (root, "maj" | "min"), or None (N, sus, power chords)
    change_counts(sequences)                 label sequences -> (2, 2, 12) counts [quality_from, quality_to, interval]
    change_matrix(counts, alpha, tau)        counts -> (25, 25) Q in majmin_labels() order, zero diagonal
    transition(Q, p_self)                    -> (25, 25) A = p_self * I + (1 - p_self) * Q
"""
import mir_eval
import numpy as np

QUALITIES = ["maj", "min"]
N_STATE = 24


def chord_class(label: str):
    """(root pitch class, "maj" or "min") of a Harte label, through its third; None if it has no third.

    Examples: "D#:maj" -> (3, "maj"), "A:min7" -> (9, "min"), "G:7" -> (7, "maj"),
    "B:hdim7" -> (11, "min") (minor third), "N" -> None, "E:sus4" -> None.
    """
    root, bitmap, _ = mir_eval.chord.encode(label)
    if root < 0:
        return None
    if bitmap[4]:
        return root, "maj"
    if bitmap[3]:
        return root, "min"
    return None


def change_counts(sequences: list[list[str]]) -> np.ndarray:
    """Count every chord change in the label sequences, by (quality_from, quality_to, root interval).

    Repeated labels and moves to the same class (e.g. D:maj -> D:maj7) are not changes; a label with
    no class breaks the sequence. Example: ["C:maj", "G:maj", "A:min"] counts [maj, maj, 7] and [maj, min, 2].
    """
    counts = np.zeros((2, 2, 12))
    for labels in sequences:
        classes = [chord_class(label) for label in labels]
        for a, b in zip(classes, classes[1:]):
            if a is None or b is None or a == b:
                continue
            counts[QUALITIES.index(a[1]), QUALITIES.index(b[1]), (b[0] - a[0]) % 12] += 1
    return counts


def change_matrix(counts: np.ndarray, alpha: float = 1.0, tau: float = 1.0) -> np.ndarray:
    """(25, 25) Q: Q[i, j] is proportional to (counts[class i -> class j] + alpha) ** tau, zero diagonal.

    State i < 12 is (i, maj), 12 <= i < 24 is (i - 12, min), 24 is N (majmin_labels() order). GuitarSet
    has no N in its labels, so moves into and out of N are uniform. tau = 0 makes every off-diagonal
    entry 1/24: the uniform Q of transition_loop.
    """
    Q = np.zeros((25, 25))
    for i in range(N_STATE):
        for j in range(N_STATE):
            if i != j:
                Q[i, j] = (counts[i // 12, j // 12, (j - i) % 12] + alpha) ** tau
    Q[:N_STATE, N_STATE] = Q[:N_STATE, :N_STATE].sum(axis=1) / (N_STATE - 1)   # chord -> N: an average move's weight
    Q[N_STATE, :N_STATE] = 1.0                                                 # N -> chord: uniform
    return Q / Q.sum(axis=1, keepdims=True)


def transition(Q: np.ndarray, p_self: float) -> np.ndarray:
    """A = p_self * I + (1 - p_self) * Q; rows sum to 1. With the uniform Q it equals transition_loop(25, p_self)."""
    return p_self * np.eye(len(Q)) + (1.0 - p_self) * Q
