"""Component 28: the listening prototype's logic, without audio or window (scripts/play_along.py runs it live).

A recording reaches the prototype as it plays (from a file read at real-time pace, or from another
application through a virtual audio cable). It is analysed block by block with V7.3's live chain and
played back to the musician `delay` seconds late, so the label of every frame is ready when that frame is
heard. This module holds what does not depend on sound devices:

    OPERATING_POINTS               V7.2's points on the C2 grid with input look-ahead 2: the decoder and
                                   the total look-ahead of each
    Analysis                       the live chain fed with blocks; per frame, the label of every point
    Reference                      an Isophonics annotation: the chord at a time and the next one
    song_list(root)                every annotated song, by album then title, for the drop-down
    chord_name(label)              "E:min7" -> "Em7", "A" -> "A", "N" -> "N.C."
    shown(...)                     what the window shows at a playback time: system chord now and next,
                                   reference chord now and next
    Onset                          the time the input first rises out of silence (the annotation's zero)

Times: input time s = samples received / sample rate. A frame f is the 2,048-sample block starting at
f * hop / sr of input time. The musician hears input time s at wall time s + delay; the window shows, at
that moment, the chords of input time s + lead.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chordtracker import online_hmm, transitions
from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS
from chordtracker.learned_emission import LABELS
from chordtracker.streaming_features import StreamingFeatures
from chordtracker.streaming_v7_2 import LookaheadStreamingCRNN

INPUT_LOOKAHEAD = 2
OPERATING_POINTS = {        # name: decoder lag (frames) and p_self; total look-ahead = 2 + 1 + lag frames
    "0.28 s": {"lag": 0, "p_self": None},       # the network alone
    "0.46 s": {"lag": 2, "p_self": 0.9},        # fixed-lag Viterbi
    "0.65 s": {"lag": 4, "p_self": 0.98},
}
QUALITIES = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "dim": "dim", "aug": "aug",
             "dim7": "dim7", "hdim7": "m7b5", "sus4": "sus4", "sus2": "sus2", "maj6": "6", "min6": "m6",
             "9": "9", "maj9": "maj9", "min9": "m9", "5": "5", "1": "(1)"}


def total_lookahead(point: str) -> int:
    """Frames between a frame and its label: input look-ahead + the harmonic part's frame + decoder lag."""
    return INPUT_LOOKAHEAD + 1 + OPERATING_POINTS[point]["lag"]


def chord_name(label: str) -> str:
    """A Harte label as a musician reads it: "E:min7" -> "Em7", "A" -> "A", "G/3" -> "G", "N" -> "N.C."."""
    if label in ("N", "X"):
        return "N.C."
    root, _, rest = label.partition(":")
    quality = rest.split("/")[0].split("(")[0] if rest else "maj"
    return root.split("/")[0] + QUALITIES.get(quality, quality)


def song_list(root: Path) -> list[tuple[str, Path]]:
    """[("01 Please Please Me - 01 I Saw Her Standing There", path), ...] for every .lab under root."""
    out = []
    for path in sorted(Path(root).rglob("*.lab")):
        album = path.parent.name.replace("_-_", " - ").replace("_", " ")
        title = path.stem.replace("_-_", " - ").replace("_", " ")
        out.append((f"{album} | {title}", path))
    return out


@dataclass
class Reference:
    """An Isophonics chord annotation: start times and labels."""
    starts: np.ndarray
    labels: list[str]

    @classmethod
    def from_lab(cls, path: Path) -> "Reference":
        rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
        return cls(np.array([float(r[0]) for r in rows]), [r[2] for r in rows])

    def at(self, t: float) -> tuple[str, str]:
        """(chord now, next different chord) at annotation time t; before the song: ("", first chord)."""
        i = int(np.searchsorted(self.starts, t, side="right")) - 1
        now = chord_name(self.labels[i]) if i >= 0 else ""
        nxt = ""
        for label in self.labels[i + 1:]:
            if chord_name(label) != now:
                nxt = chord_name(label)
                break
        return now, nxt


@dataclass
class Onset:
    """The input time at which the audio first rises above `threshold` (RMS of a block); None until then."""
    threshold: float = 1e-3
    time: float | None = None

    def update(self, block: np.ndarray, block_start: float) -> None:
        if self.time is None and len(block) and np.sqrt(np.mean(np.square(block, dtype=np.float64))) > self.threshold:
            self.time = block_start


@dataclass
class Analysis:
    """V7.3's live chain on 22,050 Hz mono blocks; labels[point][f] = the chord index of frame f, per point."""
    cfg: dict
    model: object
    prior: np.ndarray
    Q: np.ndarray
    labels: dict = field(default_factory=lambda: {p: [] for p in OPERATING_POINTS})

    def __post_init__(self):
        self.features = StreamingFeatures(self.cfg)
        self.crnn = LookaheadStreamingCRNN(self.model, INPUT_LOOKAHEAD)
        self.decoders = {p: online_hmm.FixedLagViterbi(transitions.transition(self.Q, v["p_self"]), v["lag"])
                         for p, v in OPERATING_POINTS.items() if v["p_self"] is not None}

    def step(self, block: np.ndarray) -> None:
        """Feed samples; append every label that has become known."""
        for column in self.features.step(block):
            post = self.crnn.step(column[CENTRE_ROW:CENTRE_ROW + PATCH_BINS])
            if post is None:
                continue
            for point, v in OPERATING_POINTS.items():
                if v["p_self"] is None:
                    self.labels[point].append(int(post.argmax()))
                else:
                    state = self.decoders[point].step(online_hmm.evidence(post[:, None], self.prior)[:, 0])
                    if state is not None:
                        self.labels[point].append(state)


def system_at(labels: list[int], frame: int) -> tuple[str, str]:
    """(chord of frame, next different chord already decided) from a point's labels; "" if not known yet."""
    if frame < 0 or frame >= len(labels):
        return "", ""
    now = chord_name(LABELS[labels[frame]])
    for later in labels[frame + 1:]:
        if chord_name(LABELS[later]) != now:
            return now, chord_name(LABELS[later])
    return now, ""


def shown(labels: list[int], reference: Reference | None, onset: float | None, playback_s: float, lead_s: float,
          nudge_s: float, sr: int, hop: int) -> dict:
    """What the window shows when the musician hears input time playback_s.

    The chords are those of input time playback_s + lead_s: the system's from the frame there, the
    reference's at annotation time (playback_s + lead_s - onset + nudge_s).
    """
    t = playback_s + lead_s
    frame = int(np.floor(t * sr / hop))
    system_now, system_next = system_at(labels, frame)
    ref_now = ref_next = ""
    if reference is not None and onset is not None:
        ref_now, ref_next = reference.at(t - onset + nudge_s)
    return {"system": system_now, "system_next": system_next, "reference": ref_now, "reference_next": ref_next,
            "frame": frame, "ready": frame < len(labels)}


def required_delay(point: str, lead_s: float, sr: int, hop: int, margin_frames: int = 1) -> float:
    """The smallest playback delay, s, for which every label is ready on time (one frame of margin)."""
    return (total_lookahead(point) + margin_frames) * hop / sr + lead_s
