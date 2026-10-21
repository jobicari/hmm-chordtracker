"""Component 7 (V1): beat-synchronous chroma, as plain functions.

V1 changes one thing from V0: the frame chroma is averaged per beat before the templates, so one
HMM step is one beat (about 0.5 s) instead of one frame (23 ms). Everything else (templates,
softmax, decoders, scoring) is the V0 code.

    beat_frames(path, cfg)                          audio file -> frame index of every detected beat
    segment_bounds(beats, n_frames)                 beats -> frame boundaries [0, b1, b2, ..., n_frames]
    beat_sync(chroma, bounds, aggregate)            (12, n_frames) -> (12, n_segments), one column per beat
    segments_to_annotation(path, labels, edges)     one state per beat -> timed chord segments
    take_evidence(record, extractor, emission, cfg) one take -> BeatEvidence (expensive, once)
    score_path(path, labels, evidence)              one decoded beat path -> scores for that take
    sweep_p_self(takes, labels, grid)               every p_self on cached evidence -> rows

The last three mirror chordtracker.experiment. They exist because experiment.score_path turns a
path into segments with boundaries halfway between frame times; beats have exact boundaries
(the beat times), so V1 scores with segments_to_annotation instead. choose_p_self and
write_results are reused from experiment unchanged.
"""
from dataclasses import dataclass

import librosa
import numpy as np

from chordtracker import evaluate
from chordtracker.data.data import ChordAnnotation, TakeRecord
from chordtracker.experiment import decoders_for

AGGREGATES = {"mean": np.mean, "median": np.median}


# ====================================================================================
# ================================      BEATS       ==================================
# ====================================================================================

def beat_frames(path, cfg: dict) -> np.ndarray:
    """Frame index of every beat librosa.beat.beat_track finds in the audio at path.

    Settings: cfg["audio"]["sr"] and ["mono"], cfg["features"]["hop_length"], so beat frames are on
    the chroma frame grid. cfg["beats"]["source"] must be "full": the beats come from the full signal,
    not the harmonic part the chroma uses, because HPSS removes the strums and attacks a beat tracker
    listens to (on val: mean beat F-measure 0.53 on the full signal, 0.19 on the harmonic part).
    Example: a 32 s take at 119 BPM -> about 63 beats, e.g. [22, 44, 65, ...] at hop 512.
    """
    if cfg["beats"]["source"] != "full":
        raise ValueError(f"unsupported beat source {cfg['beats']['source']!r}, expected 'full'")
    audio_cfg = cfg["audio"]
    waveform, sample_rate = librosa.load(str(path), sr=audio_cfg["sr"], mono=audio_cfg["mono"])
    _, beats = librosa.beat.beat_track(y=waveform, sr=sample_rate, hop_length=cfg["features"]["hop_length"])
    return np.asarray(beats, dtype=int)


def segment_bounds(beats: np.ndarray, n_frames: int) -> np.ndarray:
    """Frame boundaries of the beat segments: 0, every beat inside (0, n_frames), then n_frames.

    Segment k covers frames bounds[k] .. bounds[k + 1] - 1, so the stretch before the first beat and
    after the last one are segments too, and every frame is in exactly one segment.
    Example: beats [3, 7, 7, 12] with 10 frames -> [0, 3, 7, 10] (12 is past the end, the duplicate 7 is dropped)
    """
    inside = beats[(beats > 0) & (beats < n_frames)]
    return np.unique(np.concatenate([[0], inside, [n_frames]])).astype(int)


def beat_sync(chroma: np.ndarray, bounds: np.ndarray, aggregate: str = "mean") -> np.ndarray:
    """Pool the chroma frames of each segment into one column: (12, n_frames) -> (12, len(bounds) - 1).

    Column k is the mean (or median) of frames bounds[k] .. bounds[k + 1] - 1. Averaging is where the
    arpeggio notes of a beat meet in one column, and where single-frame blips are diluted.
    Example: one row [0, 1, 2, 3, 4, 5] with bounds [0, 2, 6] and "mean" -> [0.5, 3.5]
    """
    if aggregate not in AGGREGATES:
        raise ValueError(f"unsupported aggregate {aggregate!r}, expected one of {sorted(AGGREGATES)}")
    # pad=False: bounds already start at 0 and end at n_frames, so sync makes exactly one column per segment
    return librosa.util.sync(chroma, bounds.tolist(), aggregate=AGGREGATES[aggregate], pad=False)


def segments_to_annotation(path: np.ndarray, labels: list[str], edges: np.ndarray) -> ChordAnnotation:
    """Turn one state per segment into timed chord segments; segment k is [edges[k], edges[k + 1]).

    Runs of the same state merge into one chord segment, with its boundaries exactly at the edges
    (unlike evaluate.path_to_annotation, which puts boundaries halfway between frame times).
    Example with labels ["C:maj", "G:maj"] and edges [0.0, 0.5, 1.0, 1.5]:
        path [0, 0, 1]  ->  [0.0, 1.0) C:maj, [1.0, 1.5) G:maj
    """
    if len(edges) != len(path) + 1:
        raise ValueError(f"{len(path)} segments need {len(path) + 1} edges, got {len(edges)}")
    first_segments = np.concatenate([[0], np.flatnonzero(np.diff(path)) + 1])   # first segment of each run
    starts = edges[first_segments]
    ends = np.append(edges[first_segments[1:]], edges[-1])
    return ChordAnnotation(np.stack([starts, ends], axis=1), [labels[path[k]] for k in first_segments])


# ====================================================================================
# ================================     EVIDENCE     ==================================
# ====================================================================================

@dataclass(frozen=True)
class BeatEvidence:
    """What a decoder and the scoring need for one take, at beat level, computed once.

    posteriors   (n_states, n_segments), e.g. (25, 64) for a 32 s take at 119 BPM
    edges        (n_segments + 1,) seconds: segment k is [edges[k], edges[k + 1]); edges[0] = 0 and
                 edges[-1] = duration, so the segments cover the whole reference
    reference    the lead-sheet ChordAnnotation read from the take's .lab file
    duration     seconds = end of the last reference interval
    """
    song_id: str
    posteriors: np.ndarray
    edges: np.ndarray
    reference: ChordAnnotation
    duration: float


def take_evidence(record: TakeRecord, extractor, emission, cfg: dict) -> BeatEvidence:
    """Load one take: frame chroma (V0 extractor) -> beat segments -> beat chroma -> posteriors."""
    chroma = extractor.extract(record.audio)
    bounds = segment_bounds(beat_frames(record.audio, cfg), chroma.shape[1])
    reference = ChordAnnotation.from_lab(record.lab)
    duration = float(reference.intervals[-1, 1])
    edges = librosa.frames_to_time(bounds, sr=extractor.sr, hop_length=extractor.hop_length)
    edges[0], edges[-1] = 0.0, duration          # the first and last segments reach the ends of the take
    synced = beat_sync(chroma, bounds, cfg["beats"]["aggregate"])
    return BeatEvidence(song_id=record.song_id, posteriors=emission.posteriors(synced),
                        edges=edges, reference=reference, duration=duration)


# ====================================================================================
# ================================      SCORING     ==================================
# ====================================================================================

def score_path(path: np.ndarray, labels: list[str], evidence: BeatEvidence) -> tuple[evaluate.SongScore, float]:
    """Score one decoded beat path against its take's reference: (SongScore, changes per minute)."""
    estimate = segments_to_annotation(path, labels, evidence.edges)
    return evaluate.score(evidence.reference, estimate), evaluate.changes_per_minute(estimate, evidence.duration)


def sweep_p_self(takes: list[BeatEvidence], labels: list[str], grid: list[float]) -> tuple[list[dict], list[dict]]:
    """experiment.sweep_p_self at beat level: same rows, same keys, scored with the beat edges.

    Returns (summary_rows, per_song_rows) shaped like experiment.sweep_p_self, so
    experiment.choose_p_self and experiment.write_results work on them unchanged.
    """
    summary_rows, per_song_rows = [], []
    for system, decoder in decoders_for(grid).items():
        scores, changes = [], []
        for evidence in takes:
            song_score, changes_per_minute = score_path(decoder.decode(evidence.posteriors), labels, evidence)
            scores.append(song_score)
            changes.append(changes_per_minute)
            per_song_rows.append({"system": system, "song_id": evidence.song_id, "wcsr": song_score.wcsr,
                                  "changes_per_min": changes_per_minute, "total_s": song_score.total_s,
                                  "scored_s": song_score.scored_s, "matched_s": song_score.matched_s,
                                  "reduced_s": song_score.reduced_s, "excluded_s": song_score.excluded_s})
        total = evaluate.pooled(scores)
        summary_rows.append({"system": system, "wcsr": total.wcsr, "mean_changes_per_min": float(np.mean(changes)),
                             "scored_s": total.scored_s, "reduced_s": total.reduced_s,
                             "excluded_s": total.excluded_s, "n_takes": len(takes)})
    return summary_rows, per_song_rows
