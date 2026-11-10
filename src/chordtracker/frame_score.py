"""Component 34 (V10): evaluate.score for many frame paths of one song, from two precomputed tables.

A frame path turns into segments with evaluate.path_to_annotation, whose boundaries always sit halfway
between two frame centres. So frame t always owns the same stretch of time, cell(t):

    cell(0) = [0, t_0.5),  cell(t) = [t_{t-0.5}, t_{t+0.5}),  cell(T-1) = [t_{T-1.5}, duration]

whatever the path, and the score of a path is a sum over frames of what its class earns on that cell:

    matched_s(path) = sum_t M[t, path[t]],   scored_s(path) = sum_t S[t, path[t]]

M[t, c] = seconds of cell(t) (within the reference span) where the reference chord matches class c under
the metric, S[t, c] = seconds of cell(t) the metric can judge. Both come from the metric's own comparison
(evaluate.METRICS) of every reference segment with every class label, so the sums equal evaluate.score
up to float rounding. Building the tables costs one comparison per (reference segment, class); scoring
a path is then one gather, which is what makes sweeps over decoders and offsets cheap.

    tables(annotation, labels, n_frames, frame_period, metric)   -> (M, S), each (n_frames, n_classes);
                       metric: a key of evaluate.METRICS, or any mir_eval.chord comparison ('tetrads', 'mirex')
    score_path(M, S, path)                                       -> (matched_s, scored_s)
    cells(n_frames, frame_period, duration)                      -> (n_frames, 2) owned stretch per frame

Example: one reference C:maj on [0, 0.3), labels [C:maj, G:maj], frame period 0.1 s, 3 frames:
    cells [0, 0.05), [0.05, 0.15), [0.15, 0.3); path [0, 0, 1] -> matched 0.15 of 0.3 s.
"""
import mir_eval
import numpy as np

from chordtracker import evaluate
from chordtracker.data.data import ChordAnnotation


def cells(n_frames: int, frame_period: float, duration: float) -> np.ndarray:
    """(n_frames, 2) start and end of the time each frame owns in evaluate.path_to_annotation's segments."""
    mids = (np.arange(n_frames - 1) + 0.5) * frame_period
    return np.stack([np.concatenate([[0.0], mids]), np.concatenate([mids, [duration]])], axis=1)


def tables(annotation: ChordAnnotation, labels: list[str], n_frames: int, frame_period: float,
           metric: str) -> tuple[np.ndarray, np.ndarray]:
    """(M, S): matched and scored seconds of each frame's cell for each class label, as described above."""
    ref_iv = annotation.intervals
    start, end = float(ref_iv.min()), float(ref_iv.max())
    c = np.clip(cells(n_frames, frame_period, float(ref_iv[-1, 1])), start, end)
    overlap = np.clip(np.minimum(c[:, 1:2], ref_iv[None, :, 1]) - np.maximum(c[:, 0:1], ref_iv[None, :, 0]), 0, None)
    n_ref, n_cls = len(annotation.labels), len(labels)
    ref = [label for label in annotation.labels for _ in range(n_cls)]
    est = list(labels) * n_ref
    compare = evaluate.METRICS.get(metric) or getattr(mir_eval.chord, metric)    # V15: also tetrads, mirex
    comparison = compare(ref, est).reshape(n_ref, n_cls)                       # 1 right, 0 wrong, -1 not scored
    return overlap @ (comparison == 1).astype(float), overlap @ (comparison >= 0).astype(float)


def score_path(M: np.ndarray, S: np.ndarray, path: np.ndarray) -> tuple[float, float]:
    """(matched_s, scored_s) of one (n_frames,) class path."""
    t = np.arange(len(path))
    return float(M[t, path].sum()), float(S[t, path].sum())

