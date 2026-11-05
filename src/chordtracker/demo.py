"""Demo website: the chords of one prepared recording, as the page in demo/ plays them.

scripts/demo_export.py runs the listening prototype's chain (StreamingFeatures' whole-file computation,
the V7.2 network with input look-ahead 2 on the C2 grid, results/v7_2/c2_la2_seed0) on a validation
take and writes what this module builds: one JSON file per sample, read by demo/app.js. Nothing here
decodes audio; it only turns frame label paths into what the page shows.

    SYSTEMS                     the two lanes of the page: the network alone (framewise argmax) and the
                                fixed-lag Viterbi of the 0.65 s point, with the settings of play_along
    segments(path, ...)         a frame path -> [{start, end, label, name}], cut as evaluate scores it
    sample(...)                 everything one sample's JSON holds

Times are seconds of the recording. Frame t sits at t * hop / sr (acoustic_hmm.frame_times); a boundary
between two runs is halfway between their frames (evaluate.path_to_annotation), so the page shows the
same segments the WCSR is computed on. The labels are "aligned": frame t's label is the one the live
system produces for frame t, total look-ahead seconds later; with playback delayed by that look-ahead
(reports/v7_1.md, "The target set-up") it is on screen when frame t is heard.
"""
import numpy as np

from chordtracker import evaluate
from chordtracker.data.data import ChordAnnotation
from chordtracker.play_along import OPERATING_POINTS, chord_name, total_lookahead

SYSTEMS = {
    "framewise": {"title": "Network alone (framewise)", "point": "0.28 s"},
    "hmm": {"title": "HMM (fixed-lag Viterbi)", "point": "0.65 s"},
}


def segments(annotation: ChordAnnotation) -> list[dict]:
    """[{start, end, label, name}] of an annotation, times rounded to the millisecond."""
    return [{"start": round(float(s), 3), "end": round(float(e), 3), "label": label, "name": chord_name(label)}
            for (s, e), label in zip(annotation.intervals, annotation.labels)]


def path_segments(path: np.ndarray, labels: list[str], times: np.ndarray, duration: float) -> list[dict]:
    """A frame path as the page's segments: runs of one chord, boundaries halfway between frames.

    Example with labels ["C:maj", "G:maj"], frame times [0.0, 0.1, 0.2, 0.3] and duration 0.4:
        path [0, 0, 1, 1]  ->  [{0.0, 0.15, "C:maj", "C"}, {0.15, 0.4, "G:maj", "G"}]
    """
    return segments(evaluate.path_to_annotation(path, labels, times, duration))


def system_entry(key: str, path: np.ndarray, labels: list[str], times: np.ndarray, duration: float,
                 reference: ChordAnnotation, hop: int, sr: int) -> dict:
    """One lane: its settings, its segments, and its WCSR and chord changes per minute on this take."""
    point = SYSTEMS[key]["point"]
    estimate = evaluate.path_to_annotation(path, labels, times, duration)
    score = evaluate.score(reference, estimate)
    return {
        "title": SYSTEMS[key]["title"],
        "lag_frames": OPERATING_POINTS[point]["lag"],
        "p_self": OPERATING_POINTS[point]["p_self"],
        "lookahead_s": round(total_lookahead(point) * hop / sr, 3),
        "wcsr": round(100 * score.matched_s / score.scored_s, 1) if score.scored_s else None,
        "changes_per_min": round(evaluate.changes_per_minute(estimate, duration), 1),
        "segments": segments(estimate),
    }


def sample(sample_id: str, title: str, audio: str, duration: float, paths: dict[str, np.ndarray],
           labels: list[str], reference: ChordAnnotation, reference_source: str, source: dict, hop: int,
           sr: int) -> dict:
    """The JSON of one sample. paths: {"framewise": (T,), "hmm": (T,)} label indices, one per frame."""
    n = {len(p) for p in paths.values()}
    if len(n) != 1:
        raise ValueError(f"the paths have different lengths: {sorted(n)}")
    times = np.arange(n.pop()) * hop / sr
    return {
        "id": sample_id,
        "title": title,
        "audio": audio,
        "duration": round(float(duration), 3),
        "frame": {"hop": hop, "sr": sr, "seconds": round(hop / sr, 6)},
        "source": source,
        "systems": {key: system_entry(key, paths[key], labels, times, duration, reference, hop, sr) for key in SYSTEMS},
        "reference": {"source": reference_source, "segments": segments(reference)},
    }
