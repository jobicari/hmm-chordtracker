"""Demo website: the analysis result of one prepared recording, as the page in demo/ reads it.

scripts/demo_export.py runs the listening prototype's chain on a validation take and writes what this
module builds, one JSON file per sample (the spec's TrackerResult):

    duration        seconds of audio
    decoded         HMM output: fixed-lag Viterbi of the prototype's 0.65 s point   [segment, ...]
    framewise       the network alone: argmax of every frame, runs merged, nothing smoothed
    reference       the take's annotation (GuitarSet), or None
    metadata        sample rate, hop, frame count, model, decoder settings: read from the run, not typed

    segment         {"start": s, "end": s, "chord": Harte label as output, "name": as a musician reads it}

Times are seconds of the recording. Frame t sits at t * hop / sr (acoustic_hmm.frame_times). A run of
equal frames becomes one segment whose boundaries lie halfway between frames, exactly as the WCSR cuts
them (evaluate.path_to_annotation); the first starts at 0 and the last ends at the audio's duration.
Labels are aligned: frame t's label is what the live system outputs for frame t, a total look-ahead
later; with playback delayed by that look-ahead it is on screen when frame t is heard.
"""
import numpy as np

from chordtracker import evaluate
from chordtracker.data.data import ChordAnnotation
from chordtracker.play_along import OPERATING_POINTS, chord_name, total_lookahead

DECODED_POINT = "0.65 s"        # fixed-lag Viterbi, lag 4
FRAMEWISE_POINT = "0.28 s"      # the network alone


def segments(annotation: ChordAnnotation) -> list[dict]:
    """[{start, end, chord, name}] of an annotation, times rounded to the millisecond."""
    return [{"start": round(float(s), 3), "end": round(float(e), 3), "chord": label, "name": chord_name(label)}
            for (s, e), label in zip(annotation.intervals, annotation.labels)]


def path_segments(path: np.ndarray, labels: list[str], times: np.ndarray, duration: float) -> list[dict]:
    """A frame path as segments: runs of one chord, boundaries halfway between frames.

    Example with labels ["C:maj", "G:maj"], frame times [0.0, 0.1, 0.2, 0.3] and duration 0.4:
        path [0, 0, 1, 1]  ->  [{0.0, 0.15, "C:maj", "C"}, {0.15, 0.4, "G:maj", "G"}]
    """
    return segments(evaluate.path_to_annotation(path, labels, times, duration))


def result(sample_id: str, title: str, audio: str, duration: float, decoded: np.ndarray, framewise: np.ndarray,
           labels: list[str], reference: ChordAnnotation | None, reference_source: str | None, source: dict,
           metadata: dict) -> dict:
    """The JSON of one sample. decoded, framewise: (T,) label indices, one per frame of the same run."""
    if len(decoded) != len(framewise):
        raise ValueError(f"decoded has {len(decoded)} frames, framewise {len(framewise)}: not the same run")
    hop, sr = metadata["hopLength"], metadata["sampleRate"]
    times = np.arange(len(decoded)) * hop / sr
    point = OPERATING_POINTS[DECODED_POINT]
    return {
        "id": sample_id,
        "title": title,
        "audio": audio,
        "duration": round(float(duration), 3),
        "decoded": path_segments(decoded, labels, times, duration),
        "framewise": path_segments(framewise, labels, times, duration),
        "reference": segments(reference) if reference is not None else None,
        "referenceSource": reference_source if reference is not None else None,
        "metadata": {
            **metadata,
            "frameCount": int(len(decoded)),
            "frameSeconds": round(hop / sr, 6),
            "decoderName": f"fixed-lag Viterbi (lag {point['lag']} frames, p_self {point['p_self']})",
            "decodedLookaheadSeconds": round(total_lookahead(DECODED_POINT) * hop / sr, 3),
            "framewiseLookaheadSeconds": round(total_lookahead(FRAMEWISE_POINT) * hop / sr, 3),
            "vocabulary": sorted(set(labels)),
        },
        "source": source,
    }
