"""Component 12 (V5): from the CNN acoustic model's frame posteriors to V4's beat decoder, as plain functions.

The CNN outputs one posterior P(chord | frame) per CQT frame (hop 2048, ~10.8 frames/s). The decoder
is V4's, unchanged: it works on beat segments (V1 beat tracker), divides by a state prior and runs
Viterbi with the Billboard change prior. This module is only the glue between the two:

    frame_times(n_frames, sr, hop)               centre time of every CNN frame, s
    frames_to_beats(posteriors, times, edges)    (24, n_frames) -> (24, n_beats): geometric mean per beat
    training_prior(targets)                      class frequencies of the training targets in all 12 keys
    decode(beat_posteriors, Q, p_self, prior)    -> (n_beats,) state path, viterbi_discriminative as V4

Why a geometric mean over the frames of a beat (mean of log P, then renormalized)? ROADMAP §3: with
log scores, summing over a segment's frames is the segment's evidence. Averaging instead of summing
keeps one beat worth one step of the HMM whatever its length (a slow beat has ~6 frames, a fast one
~3), so p_self means the same thing as in V4. The arithmetic mean is more forgiving of one bad frame
(in a geometric mean a single frame that gives a chord ~0 vetoes it), but loses that link to the
frame-level model; it is the first thing to try on val if the geometric mean disappoints.
"""
import librosa
import numpy as np

from chordtracker import transitions

N_CLASSES = 24
EPS = 1e-8


def frame_times(n_frames: int, sr: int, hop: int) -> np.ndarray:
    """Centre time of each frame: librosa.cqt is centred, so frame t sits at t * hop / sr.

    Example: sr 22050, hop 2048 -> [0.0, 0.0929, 0.1858, ...]
    """
    return np.arange(n_frames) * hop / sr


def frames_to_beats(posteriors: np.ndarray, times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Pool (24, n_frames) frame posteriors into (24, n_beats) beat posteriors, beat k = [edges[k], edges[k + 1]).

    Each beat gets the geometric mean of the frames whose centre falls inside it, renormalized to sum
    to 1. A beat with no frame centre inside (shorter than one hop, 93 ms) takes its nearest frame.
    Example: frames [0.8, 0.2] and [0.2, 0.8] in one beat -> logs average to equal scores -> [0.5, 0.5].
    """
    logs = np.log(posteriors + EPS)
    beat_of_frame = np.searchsorted(edges, times, side="right") - 1          # -1 or n_beats: outside
    out = np.empty((posteriors.shape[0], len(edges) - 1))
    for k in range(len(edges) - 1):
        inside = beat_of_frame == k
        if inside.any():
            out[:, k] = logs[:, inside].mean(axis=1)
        else:
            out[:, k] = logs[:, np.argmin(np.abs(times - (edges[k] + edges[k + 1]) / 2))]
    out -= out.max(axis=0, keepdims=True)
    out = np.exp(out)
    return out / out.sum(axis=0, keepdims=True)


def training_prior(targets: np.ndarray) -> np.ndarray:
    """P(chord) the decoder divides by: training target frequencies with every frame in all 12 keys.

    Training sees each frame at a random shift k, so over the epochs every root is equally likely and
    only the major/minor balance of the data is left. Counting each target in all 12 transpositions
    gives that expectation exactly (as V4's training_rows did). Targets are 0..23, -1 = no target.
    Example: targets [0, 0, 12] (C:maj twice, C:min once) -> 2/36 for every major, 1/36 for every minor.
    """
    y = targets[targets >= 0]
    counts = np.zeros(N_CLASSES)
    for shift in range(12):
        counts += np.bincount((y % 12 + shift) % 12 + 12 * (y // 12), minlength=N_CLASSES)
    return counts / counts.sum()


def decode(beat_posteriors: np.ndarray, Q: np.ndarray, p_self: float, prior: np.ndarray) -> np.ndarray:
    """V4's decoder: Viterbi on the evidence P(chord | beat) / P(chord) with the transition matrix A.

    A (24 x 24) is the HMM's transition matrix: A[i, j] = P(chord j at the next beat | chord i now),
    each row sums to 1. It is built from two parts:

        A = p_self * I + (1 - p_self) * Q

        I       the 24 x 24 identity matrix, so the diagonal of A is p_self: the probability of
                staying on the same chord for one more beat. A chord then lasts 1 / (1 - p_self)
                beats on average (V4: p_self 0.5 -> 2 beats); p_self is chosen on val.
        Q       the Billboard change prior (V3, transitions.change_matrix): zero diagonal, rows sum
                to 1; Q[i, j] = P(next chord j | the chord changes, from i), counted by change type
                (e.g. major -> major up 7 semitones) in 730 Billboard songs, so it holds in every key.
        1 - p_self  the probability of a change, shared among the 23 other chords according to Q.

    Example (made-up Q value): p_self 0.5 and Q[C:maj, G:maj] 0.20 -> A[C:maj, C:maj] = 0.5,
    A[C:maj, G:maj] = 0.5 * 0.20 = 0.10, and the remaining 0.40 goes to the other 22 chords.
    Viterbi then picks the chord sequence with the highest product of evidence x A along the path.
    """
    A = transitions.transition(Q, p_self)
    return librosa.sequence.viterbi_discriminative(beat_posteriors, A, p_state=prior)
