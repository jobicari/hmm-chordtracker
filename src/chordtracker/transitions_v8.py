"""Component 33 (V8): the decoder's prior and transition matrix for a data_v8 vocabulary.

    training_prior(targets, n_classes)            P(class): train target frequencies in all 12 keys
    families(vocab)                               each quality's family: 0 = major third (or none), 1 = minor
    change_matrix(counts, vocab, prior, alpha, tau)   (n, n) Q from Billboard counts (transitions.source billboard)
    label_counts(target_sequences, n_qualities)   (n_q, n_q, 12) changes counted in frame targets
    label_change_matrix(counts, alpha, tau)       (n, n) Q from those counts (transitions.source train)
    tempered_prior(prior, power)                  prior ** power, renormalised: what the decoder divides by
    n_change_counts(target_sequences, n_chords)   (V14) changes chord -> chord, chord -> N, N -> chord
    n_rate(counts)                                (V14) share of the changes out of a chord that go to N
    add_n(Q, prior, chord_to_n)                   (V14) Q with an N row and column, (n + 1, n + 1)

The Billboard counts (results/v3/billboard_counts.json, transitions.change_counts) know only maj and
min. V8 keeps them and splits each move among the qualities of the target's family by their share
of the training prior:

    Q[i, j]  proportional to  (counts[family(i), family(j), root(j) - root(i)] + alpha) ** tau * share(quality(j))

share(q) = prior mass of q / prior mass of q's family, so C:maj -> G goes to G:maj and G:7 in the
proportion the training data holds them. Families: maj, 7, maj7, maj6, 9 and sus4 (no third, counted
with major) are family 0; min and min7 family 1. A change of quality on the same root (C:maj ->
C:7) is a change, from counts[f, f, 0], which Billboard leaves at 0: it gets alpha only.
With only maj and min in the prior this is transitions.change_matrix's chord block.

transitions.source train counts the moves in the training targets instead, by (quality_from,
quality_to, root interval), so every key shares the counts of the others:

    Q[i, j]  proportional to  (counts[quality(i), quality(j), root(j) - root(i)] + alpha) ** tau

A same-root quality change (C:maj -> C:7) then has its own count, and so has every move out of a
seventh (G:7 -> C:maj).

N (V14). With an N class after the n chord classes (data_v8.target with with_n), training_prior counts N
at its own frame share (n_chords: N has no root, so it is counted unshifted, 12 times, as every chord
is counted once per transposition), and both matrices take an N row and column (change_matrix and
label_change_matrix with chord_to_n, through add_n):

    Q'[i, N] = r                      r = chord_to_n: changes chord -> N / changes out of a chord (train labels)
    Q'[i, j] = (1 - r) * Q[i, j]      the chord moves as before
    Q'[N, j] = prior[j] / sum of the chord priors     N -> chord split by the chords' prior share
    Q'[N, N] = 0                      the diagonal stays zero; p_self is added by transitions.transition

Every row still sums to 1. A run of N frames is one N; -1 (X, chords out of vocabulary, outside the
annotation) still breaks the sequence (n_change_counts).
"""
import numpy as np

from chordtracker.data_v8 import VOCABULARIES

FAMILY = {"maj": 0, "7": 0, "maj7": 0, "maj6": 0, "9": 0, "sus4": 0, "min": 1, "min7": 1}


def training_prior(targets: np.ndarray, n_classes: int, n_chords: int | None = None) -> np.ndarray:
    """P(class) the decoder divides by: every target counted in all 12 transpositions (acoustic_hmm.training_prior).

    n_chords (V14): classes from n_chords on (N) have no root; each is counted 12 times where it is, so
    it keeps its share of the frames. Example: targets [0, 0, 36] in A (C:maj twice, C:maj7 once) -> 2/36
    for every maj, 1/36 for every maj7; [0, 60] with n_chords 60 -> 1/24 for every maj and 1/2 for N.
    """
    y = targets[targets >= 0]
    fixed = np.zeros(len(y), dtype=bool) if n_chords is None else y >= n_chords
    chords, others = y[~fixed], y[fixed]
    counts = 12 * np.bincount(others, minlength=n_classes).astype(float)
    for shift in range(12):
        counts += np.bincount((chords % 12 + shift) % 12 + 12 * (chords // 12), minlength=n_classes)
    return counts / counts.sum()


def families(vocab: str) -> np.ndarray:
    """(n_qualities,) family of each quality of the vocabulary, in class-block order."""
    return np.array([FAMILY[q] for q in VOCABULARIES[vocab]])


def change_matrix(counts: np.ndarray, vocab: str, prior: np.ndarray, alpha: float = 1.0,
                  tau: float = 1.0, chord_to_n: float | None = None) -> np.ndarray:
    """(n, n) Q over the vocabulary's classes; counts is transitions.change_counts' (2, 2, 12) array.

    chord_to_n (V14): prior has an N entry after the chords and Q gets its N row and column (add_n).
    """
    fam = families(vocab)
    if chord_to_n is not None:
        return add_n(change_matrix(counts, vocab, prior[:12 * len(fam)], alpha, tau), prior, chord_to_n)
    mass = prior.reshape(len(fam), 12).sum(axis=1)                       # prior mass per quality
    family_mass = np.array([mass[fam == f].sum() for f in (0, 1)])
    share = np.divide(mass, family_mass[fam], out=np.zeros_like(mass), where=family_mass[fam] > 0)
    n = 12 * len(fam)
    q_of, root_of = np.arange(n) // 12, np.arange(n) % 12
    Q = (counts[fam[q_of][:, None], fam[q_of][None, :], (root_of[None, :] - root_of[:, None]) % 12] + alpha) ** tau
    Q = Q * share[q_of][None, :]
    np.fill_diagonal(Q, 0.0)
    return Q / Q.sum(axis=1, keepdims=True)


def label_counts(target_sequences: list[np.ndarray], n_qualities: int) -> np.ndarray:
    """(n_q, n_q, 12) chord changes in frame targets (class = quality * 12 + root, -1 = no target).

    A run of equal frames is one chord; a frame without a target breaks the sequence, so the chords
    on either side of it are not a change. Example: [0, 0, 24, 24, 7] (C:maj, C:7, G:maj in A)
    counts [maj, 7, 0] and [7, maj, 7].
    """
    counts = np.zeros((n_qualities, n_qualities, 12))
    for y in target_sequences:
        y = np.asarray(y)
        a, b = y[:-1], y[1:]
        move = (a != b) & (a >= 0) & (b >= 0) & (a < 12 * n_qualities) & (b < 12 * n_qualities)   # N is not a chord
        np.add.at(counts, (a[move] // 12, b[move] // 12, (b[move] % 12 - a[move] % 12) % 12), 1)
    return counts


def label_change_matrix(counts: np.ndarray, alpha: float = 1.0, tau: float = 1.0, prior: np.ndarray | None = None,
                        chord_to_n: float | None = None) -> np.ndarray:
    """(n, n) Q over 12 * n_q classes from label_counts' array; zero diagonal, rows summing to 1.

    chord_to_n (V14): with the (n + 1,) prior including N, Q gets its N row and column (add_n).
    """
    if chord_to_n is not None:
        return add_n(label_change_matrix(counts, alpha, tau), prior, chord_to_n)
    n = 12 * counts.shape[0]
    q_of, root_of = np.arange(n) // 12, np.arange(n) % 12
    Q = (counts[q_of[:, None], q_of[None, :], (root_of[None, :] - root_of[:, None]) % 12] + alpha) ** tau
    np.fill_diagonal(Q, 0.0)
    return Q / Q.sum(axis=1, keepdims=True)


def tempered_prior(prior: np.ndarray, power: float = 1.0) -> np.ndarray:
    """prior ** power, summing to 1: the decoder's evidence is log P(class | frame) - log of this.

    power 1 divides out the whole training prior (V5-V7); power 0 divides by a uniform prior, so the
    evidence is the network's posterior as it is. In between, rare classes are raised less.
    Example: prior [0.75, 0.25], power 0.5 -> [0.634, 0.366].
    """
    p = np.where(prior > 0, prior, 0.0) ** power
    return p / p.sum()


def n_change_counts(target_sequences: list[np.ndarray], n_chords: int) -> dict[str, int]:
    """Changes in frame targets with N (class n_chords): chord -> chord, chord -> N and N -> chord.

    A run of equal frames is one chord (or one N); -1 breaks the sequence, as in label_counts.
    Example (n_chords 60): [0, 0, 60, 60, 7, -1, 60] -> chord_to_chord 0, chord_to_n 1, n_to_chord 1.
    """
    out = {"chord_to_chord": 0, "chord_to_n": 0, "n_to_chord": 0}
    for y in target_sequences:
        y = np.asarray(y)
        a, b = y[:-1], y[1:]
        move = (a != b) & (a >= 0) & (b >= 0)
        a_n, b_n = a[move] == n_chords, b[move] == n_chords
        out["chord_to_chord"] += int((~a_n & ~b_n).sum())
        out["chord_to_n"] += int((~a_n & b_n).sum())
        out["n_to_chord"] += int((a_n & ~b_n).sum())
    return out


def n_rate(counts: dict[str, int]) -> float:
    """Share of the changes out of a chord that go to N: chord_to_n / (chord_to_n + chord_to_chord)."""
    return counts["chord_to_n"] / (counts["chord_to_n"] + counts["chord_to_chord"])


def add_n(Q: np.ndarray, prior: np.ndarray, chord_to_n: float) -> np.ndarray:
    """(n + 1, n + 1) Q with N last: chords go to N at rate chord_to_n, N goes to chord j by its prior share.

    Q is (n, n) over the chords (zero diagonal, rows summing to 1); prior has n + 1 entries, N's last.
    """
    n = len(Q)
    out = np.zeros((n + 1, n + 1))
    out[:n, :n] = (1.0 - chord_to_n) * Q
    out[:n, n] = chord_to_n
    out[n, :n] = prior[:n] / prior[:n].sum()
    return out
