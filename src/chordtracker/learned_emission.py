"""Component 11 (V4): a learned emission, logistic regression on the V2 beat chroma, as plain functions.

V0 to V3 score each chord with a hand-made template and a softmax of sharpness 1. Those scores are
not probabilities (V3: two plausible chords differ by ~1 nat per beat), so no transition prior could
be used at its counted strength. V4 learns P(chord | beat) from the train split instead:

    x          = [treble chroma (12), bass chroma (12)] of one beat: the 24 numbers V2 already computes
    P(c | x)   = softmax(W x + b)            W (24, 24), b (24,): one row per chord, majmin order without N

Training rows are the train beats, each in all 12 transpositions (chroma and label rolled together),
so the model cannot learn train's keys. The decoder divides by the train class frequencies
(viterbi_discriminative's p_state): train is 92% major, and without the division the model's
"say major" habit costs 9 points on val (proposals/v4/ablation_1_emissions.md).
GuitarSet has no N labels, so there is no N class: V4 decodes 24 states.

    beat_features(treble, bass)            (12, n) + (12, n) -> (24, n)
    beat_targets(reference, edges)         chord class of the reference chord covering most of each beat, -1 if none
    transpose(X, y, shift)                 roll features and classes by shift semitones
    training_rows(takes)                   train BeatFeatures -> (X, y) in all 12 transpositions
    fit(X, y, C)                           -> {"W", "b", "prior"}: plain arrays, saved as JSON
    posteriors(model, X)                   (24, n) beat features -> (24, n) P(chord | beat)
    take_beats(record, cfg)                one take -> BeatFeatures (expensive, once)
    to_evidence(take, model)               BeatFeatures -> beats.BeatEvidence for the V1 scoring
"""
import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression

from chordtracker import beats, features_v2
from chordtracker.data.data import ChordAnnotation, TakeRecord
from chordtracker.templates import majmin_labels
from chordtracker.transitions import chord_class

N_CLASSES = 24
LABELS = majmin_labels()[:N_CLASSES]        # C:maj ... B:maj, C:min ... B:min (no N)


@dataclass(frozen=True)
class BeatFeatures:
    """One take at beat level, before any emission: features (24, n_beats), edges (n_beats + 1,) s."""
    song_id: str
    features: np.ndarray
    edges: np.ndarray
    reference: ChordAnnotation
    duration: float


def beat_features(treble: np.ndarray, bass: np.ndarray) -> np.ndarray:
    """Stack the treble and bass beat chroma: rows 0..11 treble C..B, rows 12..23 bass C..B."""
    return np.concatenate([treble, bass], axis=0)


def beat_targets(reference: ChordAnnotation, edges: np.ndarray) -> np.ndarray:
    """Class of the reference chord that covers most of each beat: root + 12 * is_minor, or -1.

    -1 means the beat has no maj/min class (N, sus, power chord) and is left out of training.
    Example: beat [0.0, 0.5) with C:maj on [0, 0.4) and A:min7 on [0.4, 2) -> 0 (C:maj covers 0.4 s of 0.5).
    """
    targets = np.full(len(edges) - 1, -1)
    for k, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        overlaps = np.minimum(end, reference.intervals[:, 1]) - np.maximum(start, reference.intervals[:, 0])
        if overlaps.max() <= 0:
            continue
        cls = chord_class(reference.labels[int(overlaps.argmax())])
        if cls is not None:
            targets[k] = cls[0] + 12 * (cls[1] == "min")
    return targets


def transpose(X: np.ndarray, y: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Move beats up by shift semitones: roll treble and bass chroma, move each root, keep the quality.

    Example: shift 2 turns a C:maj beat (class 0) with energy on C, E, G into a D:maj beat (class 2)
    with energy on D, F#, A. Class -1 stays -1.
    """
    X_up = np.concatenate([np.roll(X[:12], shift, axis=0), np.roll(X[12:], shift, axis=0)])
    y_up = np.where(y >= 0, (y % 12 + shift) % 12 + 12 * (y // 12), -1)
    return X_up, y_up


def training_rows(takes: list[BeatFeatures]) -> tuple[np.ndarray, np.ndarray]:
    """Every labelled beat of the takes in all 12 transpositions: X (n_rows, 24), y (n_rows,).

    On GuitarSet train: 3,294 labelled beats -> 39,528 rows.
    """
    rows, classes = [], []
    for take in takes:
        y = beat_targets(take.reference, take.edges)
        for shift in range(12):
            X_up, y_up = transpose(take.features, y, shift)
            rows.append(X_up[:, y_up >= 0].T)
            classes.append(y_up[y_up >= 0])
    return np.concatenate(rows), np.concatenate(classes)


def fit(X: np.ndarray, y: np.ndarray, C: float) -> dict:
    """Multinomial logistic regression (L2, strength 1 / C) -> {"W" (24, 24), "b" (24,), "prior" (24,)}.

    prior is the class frequency in y, the p_state the decoder divides by. Every class must occur in y
    (the 12 transpositions see to that).
    """
    if len(np.unique(y)) != N_CLASSES:
        raise ValueError(f"y has {len(np.unique(y))} classes, need all {N_CLASSES}")
    regression = LogisticRegression(C=C, max_iter=3000).fit(X, y)
    return {"W": regression.coef_, "b": regression.intercept_,
            "prior": np.bincount(y, minlength=N_CLASSES) / len(y)}


def posteriors(model: dict, X: np.ndarray) -> np.ndarray:
    """softmax(W @ X + b) per column: (24, n_beats) features -> (24, n_beats), columns sum to 1."""
    scores = model["W"] @ X + model["b"][:, None]
    scores -= scores.max(axis=0, keepdims=True)
    unnormalized = np.exp(scores)
    return unnormalized / unnormalized.sum(axis=0, keepdims=True)


def save(model: dict, path: Path) -> None:
    """Write the fitted arrays as JSON, so a test run reads them instead of refitting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: np.asarray(value).tolist() for key, value in model.items()}))


def load(path: Path) -> dict:
    return {key: np.array(value) for key, value in json.loads(path.read_text()).items()}


def take_beats(record: TakeRecord, cfg: dict) -> BeatFeatures:
    """One take: V2 treble and bass frame chroma -> V1 beat segments -> beat means (as bass.take_evidence)."""
    treble, bass = features_v2.extract(record.audio, cfg)
    bounds = beats.segment_bounds(beats.beat_frames(record.audio, cfg), treble.shape[1])
    reference = ChordAnnotation.from_lab(record.lab)
    duration = float(reference.intervals[-1, 1])
    edges = librosa.frames_to_time(bounds, sr=cfg["audio"]["sr"], hop_length=cfg["features"]["hop_length"])
    edges[0], edges[-1] = 0.0, duration
    aggregate = cfg["beats"]["aggregate"]
    return BeatFeatures(record.song_id, beat_features(beats.beat_sync(treble, bounds, aggregate),
                                                      beats.beat_sync(bass, bounds, aggregate)),
                        edges, reference, duration)


def to_evidence(take: BeatFeatures, model: dict) -> beats.BeatEvidence:
    """The learned posteriors of one take, in the BeatEvidence the V1 scoring reads."""
    return beats.BeatEvidence(take.song_id, posteriors(model, take.features), take.edges, take.reference,
                              take.duration)
