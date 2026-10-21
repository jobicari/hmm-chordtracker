"""Component 9 (V2): the bass term of the emission, and V2 take evidence, as plain functions.

    score(chord) = template . treble_chroma + w * (root_onehot . bass_chroma)
    posteriors   = softmax(beta * score) per column

The treble part is the V0 TemplateEmission score, unchanged. The bass term adds w times the bass
chroma value of the chord's root, so between two triads that share two notes (D:min = D F A,
F:maj = F A C) the one whose root is in the bass wins. N gets w * no_chord_value * sum(bass),
the same rule as its treble template.

    root_matrix(no_chord_value)                       (25, 12): 1 on each chord's root, N row flat
    posteriors(model, treble, bass, w)                beat chromas -> (25, n_beats) posteriors
    take_evidence(record, model, cfg)                 one take -> beats.BeatEvidence (expensive, once)
"""
import librosa
import numpy as np

from chordtracker import beats, features_v2
from chordtracker.data.data import ChordAnnotation, TakeRecord
from chordtracker.emission import TemplateEmission


def root_matrix(no_chord_value: float = 0.25) -> np.ndarray:
    """(25, 12) in majmin_labels() order: row i has 1 on chord i's root; the N row is no_chord_value everywhere.

    Example: row 2 (D:maj) and row 14 (D:min) both have their 1 in column 2 (D).
    """
    roots = np.zeros((25, 12))
    for root in range(12):
        roots[root, root] = roots[root + 12, root] = 1.0
    roots[24] = no_chord_value
    return roots


def posteriors(model: TemplateEmission, treble: np.ndarray, bass: np.ndarray, w: float) -> np.ndarray:
    """softmax(beta * (templates @ treble + w * roots @ bass)) per column; w = 0 is model.posteriors(treble)."""
    roots = root_matrix(model.templates[24, 0])
    scaled = model.beta * (model.scores(treble) + w * roots @ bass)
    scaled -= scaled.max(axis=0, keepdims=True)      # as TemplateEmission: no overflow, same result
    unnormalized = np.exp(scaled)
    return unnormalized / unnormalized.sum(axis=0, keepdims=True)


def take_evidence(record: TakeRecord, model: TemplateEmission, cfg: dict) -> beats.BeatEvidence:
    """One take: V2 treble and bass chroma -> V1 beat segments -> beat means -> posteriors with the bass term."""
    treble, bass = features_v2.extract(record.audio, cfg)
    bounds = beats.segment_bounds(beats.beat_frames(record.audio, cfg), treble.shape[1])
    reference = ChordAnnotation.from_lab(record.lab)
    duration = float(reference.intervals[-1, 1])
    edges = librosa.frames_to_time(bounds, sr=cfg["audio"]["sr"], hop_length=cfg["features"]["hop_length"])
    edges[0], edges[-1] = 0.0, duration          # the first and last segments reach the ends of the take
    aggregate = cfg["beats"]["aggregate"]
    beat_posteriors = posteriors(model, beats.beat_sync(treble, bounds, aggregate),
                                 beats.beat_sync(bass, bounds, aggregate), cfg["bass"]["weight"])
    return beats.BeatEvidence(song_id=record.song_id, posteriors=beat_posteriors, edges=edges,
                              reference=reference, duration=duration)
