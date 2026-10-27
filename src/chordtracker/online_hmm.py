"""Component 17 (V7): HMM decoders that only use past frames, for real-time chord recognition.

V5 and V6 decode with Viterbi over the whole song: the backtrace starts at the last beat, so no chord
is final before the song ends. V7 decodes per CRNN frame (93 ms, no beats) with decoders whose
decision for frame t only reads frames up to t + L, for a lag L chosen by the user:

    evidence(posteriors, prior)                 (24, T) P(chord | frame) -> log P(frame | chord) + const
    ForwardFilter(A).step(log_b)                 argmax of P(chord at t | frames 0..t), lag 0
    FixedLagViterbi(A, lag).step(log_b)          the best path over frames 0..t, read at t - lag
    forward_filter(posteriors, A, prior)         (T,) path, ForwardFilter over a whole song
    fixed_lag_viterbi(posteriors, A, prior, lags) {lag: (T,) path} for several lags in one pass

Symbols: t is a frame index, T the number of frames, L (lag) the number of frames the decoder waits
before it decides on a frame, A the 24 x 24 transition matrix (acoustic_hmm.decode), b_t(j) the
evidence for chord j at frame t: P(chord j | frame t) / P(chord j), the posterior divided by the
training prior, as viterbi_discriminative does (ROADMAP section 3).

Forward filtering (sum over paths). alpha_t(j) = P(chord j at t | frames 0..t), updated per frame:

    alpha_t(j)  proportional to  b_t(j) * sum_i alpha_{t-1}(i) A[i, j]

The decision for frame t is argmax_j alpha_t(j). It never changes its mind about a frame, but it can
only use the evidence up to that frame.

Fixed-lag Viterbi (max over paths). delta_t(j) = log-probability of the best path that ends in chord
j at frame t, psi_t(j) = the chord at t - 1 on that path (the backpointer):

    delta_t(j) = log b_t(j) + max_i [delta_{t-1}(i) + log A[i, j]],   psi_t(j) = argmax_i [...]

At frame t the decoder follows the backpointers from argmax_j delta_t(j) down to frame t - L and
outputs that chord for frame t - L. With L = 0 this is the max-product twin of forward filtering; as
L grows to the song length it is offline Viterbi, decision for decision. At the end of a song the
last L frames are read off the final best path (`flush`), as a streaming system would when the music stops.
"""
from collections import deque

import numpy as np

EPS = 1e-10


def evidence(posteriors: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """(24, T) posteriors and (24,) prior -> (24, T) log b_t(j) = log P(j | frame t) - log P(j).

    The same scaled likelihood as librosa's viterbi_discriminative. Example: posterior [0.6, 0.4],
    prior [0.75, 0.25] -> log 0.8 and log 1.6: the rarer chord gets the higher evidence.
    """
    return np.log(posteriors + EPS) - np.log(prior + EPS)[:, None]


class ForwardFilter:
    """Lag 0, sum over paths: one step per frame returns argmax_j P(chord j at t | frames 0..t)."""

    def __init__(self, A: np.ndarray):
        self.A = A
        self.alpha = None

    def step(self, log_b: np.ndarray) -> int:
        """Read frame t's (24,) log evidence, return the chord decided for frame t."""
        b = np.exp(log_b - log_b.max())
        predicted = np.full(len(b), 1.0 / len(b)) if self.alpha is None else self.alpha @ self.A
        alpha = predicted * b
        self.alpha = alpha / alpha.sum()
        return int(self.alpha.argmax())


class FixedLagViterbi:
    """Max over paths with a lag of L frames: step(frame t) returns the chord decided for frame t - L.

    Keeps delta_t (24,) and the last L backpointer rows, so memory and work per frame are O(24 x 24 + L).
    The first L steps return None (no frame is old enough yet); flush() returns the last L decisions.
    """

    def __init__(self, A: np.ndarray, lag: int):
        self.log_A = np.log(A + EPS)
        self.lag = lag
        self.delta = None
        self.psi = deque(maxlen=max(lag, 1))
        self.t = -1

    def step(self, log_b: np.ndarray) -> int | None:
        """Read frame t's (24,) log evidence, return the chord decided for frame t - L (None while t < L)."""
        self.t += 1
        if self.delta is None:
            self.delta = log_b - np.log(len(log_b))                    # uniform start, as librosa
        else:
            scores = self.delta[:, None] + self.log_A                  # [i, j]: from chord i to chord j
            psi = scores.argmax(axis=0)
            self.delta = scores[psi, np.arange(len(psi))] + log_b
            self.delta -= self.delta.max()                             # only differences matter
            if self.lag:
                self.psi.append(psi)
        if self.t < self.lag:
            return None
        state = int(self.delta.argmax())
        for psi in reversed(self.psi) if self.lag else ():           # psi_t, psi_{t-1}, ..., psi_{t-L+1}
            state = int(psi[state])
        return state

    def flush(self) -> list[int]:
        """At the end of the song: the decisions for the last min(L, t + 1) frames, from the final best path."""
        n = min(self.lag, self.t + 1)
        if n == 0:
            return []
        path = [int(self.delta.argmax())]
        for psi in list(reversed(self.psi))[:n - 1]:
            path.append(int(psi[path[-1]]))
        return path[::-1]


def forward_filter(posteriors: np.ndarray, A: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """(24, T) posteriors -> (T,) chords, each decided from frames 0..t only."""
    f = ForwardFilter(A)
    return np.array([f.step(log_b) for log_b in evidence(posteriors, prior).T], dtype=int)


def fixed_lag_viterbi(posteriors: np.ndarray, A: np.ndarray, prior: np.ndarray,
                      lags: list[int]) -> dict[int, np.ndarray]:
    """(24, T) posteriors -> {L: (T,) chords} for every lag, the same decisions FixedLagViterbi makes.

    One forward pass stores every delta_t and psi_t; then, for each L, all frames are backtracked at
    once (L vectorized steps instead of T separate backtraces). Frame t's chord comes from the best
    path at frame min(t + L, T - 1), so the last L frames are the flush of the streaming decoder.
    """
    log_b = evidence(posteriors, prior)
    log_A = np.log(A + EPS)
    n, T = log_b.shape
    delta = np.empty((T, n))
    psi = np.zeros((T, n), dtype=int)
    delta[0] = log_b[:, 0] - np.log(n)
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores[psi[t], np.arange(n)] + log_b[:, t]
        delta[t] -= delta[t].max()
    best = delta.argmax(axis=1)                                        # end of the best path at each frame
    frames = np.arange(T)
    paths = {}
    for lag in lags:
        read_at = np.minimum(frames + lag, T - 1)                      # the frame at which t is decided
        state = best[read_at]
        for k in range(min(lag, T - 1)):
            step_back = read_at - k > frames                            # still above frame t: follow psi
            state = np.where(step_back, psi[np.maximum(read_at - k, 0), state], state)
        paths[lag] = state
    return paths
