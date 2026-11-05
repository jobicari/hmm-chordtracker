"""Demo website: the analysis result of one prepared recording, as the page in demo/ reads it.

scripts/demo_export.py runs the listening prototype's chain on a validation take and writes what this
module builds, one JSON file per sample (the spec's TrackerResult):

    duration        seconds of audio
    points          the prototype's three operating points (play_along.OPERATING_POINTS): the network
                    alone (0.28 s), fixed-lag Viterbi lag 2 (0.46 s) and lag 4 (0.65 s); each with its
                    segments and the delay after which a frame's label is known
    framewise       the network alone: argmax of every frame, runs merged, nothing smoothed
    reference       the take's annotation (GuitarSet), or None
    metadata        sample rate, hop, frame count, model, decoder settings: read from the run, not typed

    segment         {"start": s, "end": s, "chord": Harte label as output, "name": as a musician reads it}

Times are seconds of the recording. Frame t sits at t * hop / sr (acoustic_hmm.frame_times). A run of
equal frames becomes one segment whose boundaries lie halfway between frames, exactly as the WCSR cuts
them (evaluate.path_to_annotation); the first starts at 0 and the last ends at the audio's duration.
Labels are aligned: frame t's label is what the live system outputs for frame t, known at input time
t * hop / sr + decisionDelaySeconds and never changed afterwards (a fixed-lag decision is final), so
revealing each label at that time shows exactly what a live decoder would have shown.
"""
import numpy as np

from chordtracker import evaluate
from chordtracker.data.data import ChordAnnotation
from chordtracker.play_along import OPERATING_POINTS, chord_name, required_delay, total_lookahead

DEFAULT_POINT = "0.65 s"        # fixed-lag Viterbi, lag 4: the steadiest point by ear (reports/play_along.md)
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


def decoder_name(point: str) -> str:
    """"0.65 s" -> "fixed-lag Viterbi (lag 4 frames, p_self 0.98)"; the 0.28 s point is the network alone."""
    v = OPERATING_POINTS[point]
    if v["p_self"] is None:
        return "network alone (no decoder look-ahead)"
    return f"fixed-lag Viterbi (lag {v['lag']} frames, p_self {v['p_self']})"


def playback_delay(point: str, sr: int, hop: int) -> float:
    """Playback delay for which every label is on screen when its time is heard.

    Frame f is shown from (f - 1/2) h (its segment starts halfway from the previous frame) but decided
    only at input time f h + the decision delay, so playback waits half a frame more than that delay.
    At 0.65 s: (7 + 1) frames + 1/2 frame = 0.789 s.
    """
    return required_delay(point, 0.0, sr, hop) + 0.5 * hop / sr


def result(sample_id: str, title: str, audio: str, duration: float, points: dict[str, np.ndarray],
           framewise: np.ndarray, labels: list[str], reference: ChordAnnotation | None, reference_source: str | None,
           source: dict, metadata: dict) -> dict:
    """The JSON of one sample.

    points: {"0.28 s": path, "0.46 s": path, "0.65 s": path}, the label path of every operating point;
    framewise: the argmax path. All (T,) label indices of the same run. Each point also gets the delay
    after which a frame's label is known (play_along.required_delay: total look-ahead + one frame), so
    the page can reveal every label when a live decoder would have output it, and the playback delay
    that puts every label on screen in time (playback_delay).
    """
    lengths = {len(framewise)} | {len(p) for p in points.values()}
    if len(lengths) != 1:
        raise ValueError(f"the paths have different lengths {sorted(lengths)}: not the same run")
    if set(points) != set(OPERATING_POINTS):
        raise ValueError(f"expected the points {sorted(OPERATING_POINTS)}, got {sorted(points)}")
    hop, sr = metadata["hopLength"], metadata["sampleRate"]
    times = np.arange(len(framewise)) * hop / sr
    return {
        "id": sample_id,
        "title": title,
        "audio": audio,
        "duration": round(float(duration), 3),
        "points": [{
            "id": point,
            "decoder": decoder_name(point),
            "lagFrames": OPERATING_POINTS[point]["lag"],
            "pSelf": OPERATING_POINTS[point]["p_self"],
            "lookaheadSeconds": round(total_lookahead(point) * hop / sr, 3),
            "decisionDelaySeconds": round(required_delay(point, 0.0, sr, hop), 3),
            "playbackDelaySeconds": round(playback_delay(point, sr, hop), 3),
            "segments": path_segments(points[point], labels, times, duration),
        } for point in OPERATING_POINTS],
        "defaultPoint": DEFAULT_POINT,
        "framewise": path_segments(framewise, labels, times, duration),
        "framewiseDecisionDelaySeconds": round(required_delay(FRAMEWISE_POINT, 0.0, sr, hop), 3),
        "reference": segments(reference) if reference is not None else None,
        "referenceSource": reference_source if reference is not None else None,
        "metadata": {**metadata, "frameCount": int(len(framewise)), "frameSeconds": round(hop / sr, 6),
                     "vocabulary": sorted(set(labels))},
        "source": source,
    }
