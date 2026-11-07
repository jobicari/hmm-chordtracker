"""Component 35 (V10): changes to the network's posteriors and to the decoded path, before and after decoding.

V8's decoders (online_hmm) are unchanged. V10 changes what goes into them and what comes out:

    ensemble(posteriors, mode)          one (n, T) posterior from several models' (n, T) posteriors
    quality_offsets(posterior, offsets) one additive log offset per quality block, renormalised
    coordinate_search(objective, start, grid, free)   offsets fitted one coordinate at a time
    level_gate(path, level, threshold, n_label)       N wherever the frame's input level is below a threshold
    decode_points(posterior, A_of, prior, lags)       the V8 points: network alone and fixed-lag Viterbi

Ensemble. Arithmetic: the mean of the posteriors. Geometric: exp of the mean log posterior, renormalised
per frame (a product of experts: a class must be likely under every model). Example, two models on two
classes, [0.9, 0.1] and [0.5, 0.5]: arithmetic [0.7, 0.3], geometric [0.75, 0.25].

Quality offsets. V8 divides the posterior by prior ** 0.25: one exponent for all classes. The offsets give
each quality block of 12 classes its own log factor, the same for every root:

    posterior'(q * 12 + r, t)  proportional to  posterior(q * 12 + r, t) * exp(offset[q])

with offset[maj] = 0 (only differences between qualities matter). The decoder then divides by the
tempered prior as before, so offsets of 0 are V8. Example: offsets [0, 0, log 2] on qualities (maj, min,
7) double every 7 class before the frame is renormalised.

Level gate. The network always names a chord. The gate replaces the decision of frame t by N (class
n_label) when level[t] < threshold, after decoding, so the chord path elsewhere is untouched.
"""
from collections.abc import Callable

import numpy as np

from chordtracker import online_hmm

EPS = 1e-10


def ensemble(posteriors: list[np.ndarray], mode: str) -> np.ndarray:
    """(n, T) combined posterior of several (n, T) posteriors; mode 'arithmetic' or 'geometric'."""
    stack = np.stack(posteriors)
    if mode == "arithmetic":
        return stack.mean(axis=0)
    if mode == "geometric":
        log_mean = np.log(stack + EPS).mean(axis=0)
        p = np.exp(log_mean - log_mean.max(axis=0, keepdims=True))
        return p / p.sum(axis=0, keepdims=True)
    raise ValueError(f"unknown ensemble mode {mode!r}")


def quality_offsets(posterior: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """(n, T) posterior times exp(offsets[class // 12]), renormalised per frame; offsets is (n // 12,)."""
    w = np.exp(np.repeat(np.asarray(offsets, dtype=float), 12))[:, None]
    p = posterior * w
    return p / p.sum(axis=0, keepdims=True)


def coordinate_search(objective: Callable[[np.ndarray], float], start: np.ndarray, grid: list[float],
                      free: list[int], max_rounds: int = 4) -> tuple[np.ndarray, float, list[dict]]:
    """Maximise objective(offsets) one free coordinate at a time over the grid, until a round changes nothing.

    Each round tries every grid value for each free coordinate in turn, the others held, and keeps the best
    (ties keep the current value). Returns the best offsets, their objective, and one row per evaluation.
    """
    x = np.array(start, dtype=float)
    best = objective(x)
    trace = [{"round": 0, "offsets": x.tolist(), "objective": best}]
    for round_ in range(1, max_rounds + 1):
        changed = False
        for i in free:
            for value in grid:
                if value == x[i]:
                    continue
                y = x.copy()
                y[i] = value
                f = objective(y)
                trace.append({"round": round_, "offsets": y.tolist(), "objective": f})
                if f > best + 1e-12:
                    best, x, changed = f, y, True
        if not changed:
            break
    return x, best, trace


def level_gate(path: np.ndarray, level: np.ndarray, threshold: float, n_label: int) -> np.ndarray:
    """The path with n_label (N) on every frame whose level is below threshold."""
    return np.where(level < threshold, n_label, path)


def decode_points(posterior: np.ndarray, A_of: dict[float, np.ndarray], prior: np.ndarray,
                  lags: list[int]) -> dict[tuple[str, float | None], np.ndarray]:
    """{(point, p_self): (T,) path}: ('network', None) is the frame argmax; ('lag L', p) fixed-lag Viterbi
    at lag L with the transition matrix A_of[p], the evidence divided by prior (online_hmm.evidence)."""
    out = {("network", None): posterior.argmax(axis=0)}
    for p_self, A in A_of.items():
        for lag, path in online_hmm.fixed_lag_viterbi(posterior, A, prior, lags).items():
            out[(f"lag {lag}", p_self)] = path
    return out
