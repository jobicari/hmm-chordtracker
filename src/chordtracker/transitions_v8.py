"""Component 33 (V8): the decoder's prior and transition matrix for a data_v8 vocabulary.

    training_prior(targets, n_classes)            P(class): train target frequencies in all 12 keys
    families(vocab)                               each quality's family: 0 = major third (or none), 1 = minor
    change_matrix(counts, vocab, prior, alpha, tau)   (n, n) Q from Billboard counts (transitions.source billboard)
    label_counts(target_sequences, n_qualities)   (n_q, n_q, 12) changes counted in frame targets
    label_change_matrix(counts, alpha, tau)       (n, n) Q from those counts (transitions.source train)
    tempered_prior(prior, power)                  prior ** power, renormalised: what the decoder divides by

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
"""
import numpy as np

from chordtracker.data_v8 import VOCABULARIES

FAMILY = {"maj": 0, "7": 0, "maj7": 0, "maj6": 0, "9": 0, "sus4": 0, "min": 1, "min7": 1}


def training_prior(targets: np.ndarray, n_classes: int) -> np.ndarray:
    """P(class) the decoder divides by: every target counted in all 12 transpositions (acoustic_hmm.training_prior).

    Example: targets [0, 0, 36] in A (C:maj twice, C:maj7 once) -> 2/36 for every maj, 1/36 for every maj7.
    """
    y = targets[targets >= 0]
    counts = np.zeros(n_classes)
    for shift in range(12):
        counts += np.bincount((y % 12 + shift) % 12 + 12 * (y // 12), minlength=n_classes)
    return counts / counts.sum()


def families(vocab: str) -> np.ndarray:
    """(n_qualities,) family of each quality of the vocabulary, in class-block order."""
    return np.array([FAMILY[q] for q in VOCABULARIES[vocab]])


def change_matrix(counts: np.ndarray, vocab: str, prior: np.ndarray, alpha: float = 1.0,
                  tau: float = 1.0) -> np.ndarray:
    """(n, n) Q over the vocabulary's classes; counts is transitions.change_counts' (2, 2, 12) array."""
    fam = families(vocab)
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
        move = (a != b) & (a >= 0) & (b >= 0)
        np.add.at(counts, (a[move] // 12, b[move] // 12, (b[move] % 12 - a[move] % 12) % 12), 1)
    return counts


def label_change_matrix(counts: np.ndarray, alpha: float = 1.0, tau: float = 1.0) -> np.ndarray:
    """(n, n) Q over 12 * n_q classes from label_counts' array; zero diagonal, rows summing to 1."""
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
