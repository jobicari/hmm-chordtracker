"""The listening prototype's logic, without audio or window (scripts/play_along.py runs it live).

A recording reaches the prototype as it plays (from a file read at real-time pace, or from another
application through a virtual audio cable). It is analysed block by block with the live chain and the
frozen V11 student (vocabulary A: maj, min, 7, maj7, min7, 60 classes; configs/v11.yaml `frozen`), and
played back to the musician `delay` seconds late, so the label of every frame is ready when that frame
is heard. This module holds what does not depend on sound devices:

    OPERATING_POINTS               V11's four frozen points (input look-ahead 2): the decoder lag and
                                   p_self of each, and so its total look-ahead
    V7_2_POINTS                    the demo website's three points (V7.2's network and p_self)
    CLASS_LABELS                   the Harte label of each of the 60 classes ('C:maj' .. 'B:min7')
    Analysis                       the live chain fed with blocks; per frame, the label of every point;
                                   finish() at the end of a recording
    Reference                      an Isophonics annotation, every chord reduced to vocabulary A (the
                                   sevenths mapping): the chord at a time and the next one
    reference_name(label)          "D:7(b9)" -> "D7", "C:maj6" -> "C", "B:dim" -> "(Bdim)" (not in the vocabulary)
    song_list(root)                every annotated song, by album then title, for the drop-down
    chord_name(label)              "E:min7" -> "Em7", "G:7" -> "G7", "A" -> "A", "N" -> "N.C."
    shown(...)                     what the window shows at a playback time: system chord now and next,
                                   reference chord now and next
    held(now, last)                the last labelled system chord, kept while the current frame is not
                                   labelled yet (the tuning step at 10 s can hold the analysis ~0.4 s)
    Onset                          the time the input first rises out of silence (the annotation's zero)

The decoders are V8's: fixed-lag Viterbi on log P(class | frame) - log(prior ** prior_power), with the
Billboard change matrix Q and the point's p_self on the diagonal.

Times: input time s = samples received / sample rate. A frame f is the 2,048-sample block starting at
f * hop / sr of input time. The musician hears input time s at wall time s + delay; the window shows, at
that moment, the chords of input time s + lead.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chordtracker import data_v8, online_hmm, transitions
from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS
from chordtracker.streaming_features import StreamingFeatures
from chordtracker.streaming_v7_2 import LookaheadStreamingCRNN

VOCAB = "A"
CLASS_LABELS = data_v8.class_names(VOCAB)
INPUT_LOOKAHEAD = 2
OPERATING_POINTS = {        # name (total look-ahead): decoder lag (frames) and p_self, as configs/v11.yaml frozen
    "0.28 s": {"lag": 0, "p_self": None},       # the network alone
    "0.46 s": {"lag": 2, "p_self": 0.3},        # fixed-lag Viterbi
    "0.65 s": {"lag": 4, "p_self": 0.7},
    "1.02 s": {"lag": 8, "p_self": 0.98},
}
V7_2_POINTS = {             # the demo website's points: V7.2's 24-class network (demo_server.py, demo_export.py)
    "0.28 s": {"lag": 0, "p_self": None},
    "0.46 s": {"lag": 2, "p_self": 0.9},
    "0.65 s": {"lag": 4, "p_self": 0.98},
}
QUALITIES = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "dim": "dim", "aug": "aug",
             "dim7": "dim7", "hdim7": "m7b5", "sus4": "sus4", "sus2": "sus2", "maj6": "6", "min6": "m6",
             "9": "9", "maj9": "maj9", "min9": "m9", "5": "5", "1": "(1)"}


def total_lookahead(point: str) -> int:
    """Frames between a frame and its label: input look-ahead + the harmonic part's frame + decoder lag.

    A point's name is its total look-ahead, so a name has the same lag in OPERATING_POINTS and V7_2_POINTS.
    """
    return INPUT_LOOKAHEAD + 1 + {**V7_2_POINTS, **OPERATING_POINTS}[point]["lag"]


def chord_name(label: str) -> str:
    """A Harte label as a musician reads it: "E:min7" -> "Em7", "A" -> "A", "G/3" -> "G", "N" -> "N.C."."""
    if label in ("N", "X"):
        return "N.C."
    root, _, rest = label.partition(":")
    quality = rest.split("/")[0].split("(")[0] if rest else "maj"
    return root.split("/")[0] + QUALITIES.get(quality, quality)


def reference_name(label: str) -> str:
    """An annotation label as vocabulary A scores it (data_v8.target, mir_eval's sevenths): the class's name;
    "N.C." for N and X; a chord outside the vocabulary (dim, sus4, aug, ...) in parentheses, as it is."""
    if label in ("N", "X"):
        return "N.C."
    target = data_v8.target(label, VOCAB)
    return chord_name(CLASS_LABELS[target]) if target >= 0 else f"({chord_name(label)})"


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
    """An Isophonics chord annotation: start times and labels, shown reduced to vocabulary A."""
    starts: np.ndarray
    labels: list[str]

    @classmethod
    def from_lab(cls, path: Path) -> "Reference":
        rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
        return cls(np.array([float(r[0]) for r in rows]), [r[2] for r in rows])

    def at(self, t: float) -> tuple[str, str]:
        """(chord now, next different chord) at annotation time t; before the song: ("", first chord)."""
        i = int(np.searchsorted(self.starts, t, side="right")) - 1
        now = reference_name(self.labels[i]) if i >= 0 else ""
        nxt = ""
        for label in self.labels[i + 1:]:
            if reference_name(label) != now:
                nxt = reference_name(label)
                break
        return now, nxt


@dataclass
class Onset:
    """The input time at which the audio first rises above `threshold` (RMS of a block); None until then.

    A paused player sends digital silence, so the threshold is low (-100 dBFS): any music starts the clock.
    """
    threshold: float = 1e-5
    time: float | None = None

    def update(self, block: np.ndarray, block_start: float) -> None:
        if self.time is None and len(block) and np.sqrt(np.mean(np.square(block, dtype=np.float64))) > self.threshold:
            self.time = block_start


@dataclass
class Analysis:
    """The live chain on 22,050 Hz mono blocks; labels[point][f] = the class index of frame f, per point.

    prior: what the decoders divide by (transitions_v8.tempered_prior of the training prior); Q: the change
    matrix (zero diagonal) that each point's p_self completes; points: OPERATING_POINTS or V7_2_POINTS.
    """
    cfg: dict
    model: object
    prior: np.ndarray
    Q: np.ndarray
    points: dict = field(default_factory=lambda: OPERATING_POINTS)

    def __post_init__(self):
        self.labels = {p: [] for p in self.points}
        self.features = StreamingFeatures(self.cfg)
        self.crnn = LookaheadStreamingCRNN(self.model, INPUT_LOOKAHEAD)
        self.decoders = {p: online_hmm.FixedLagViterbi(transitions.transition(self.Q, v["p_self"]), v["lag"])
                         for p, v in self.points.items() if v["p_self"] is not None}

    def step(self, block: np.ndarray) -> None:
        """Feed samples; append every label that has become known."""
        for column in self.features.step(block):
            self._column(column)

    def finish(self) -> None:
        """The recording has ended: compute the remaining columns, frames and decisions (silence after the end).

        Afterwards every point has one label per frame, the same as the whole-file computation.
        """
        for column in self.features.flush():
            self._column(column)
        for post in self.crnn.flush():
            self._posterior(post)
        for point, decoder in self.decoders.items():
            self.labels[point].extend(decoder.flush())

    def _column(self, column: np.ndarray) -> None:
        post = self.crnn.step(column[CENTRE_ROW:CENTRE_ROW + PATCH_BINS])
        if post is not None:
            self._posterior(post)

    def _posterior(self, post: np.ndarray) -> None:
        for point, v in self.points.items():
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
    now = chord_name(CLASS_LABELS[labels[frame]])
    for later in labels[frame + 1:]:
        if chord_name(CLASS_LABELS[later]) != now:
            return now, chord_name(CLASS_LABELS[later])
    return now, ""


def shown(labels: list[int], reference: Reference | None, onset: float | None, playback_s: float, lead_s: float,
          nudge_s: float, sr: int, hop: int) -> dict:
    """What the window shows when the musician hears input time playback_s.

    The chords are those of input time playback_s + lead_s. The analysis starts at the onset, so the
    system's frame is counted from it, as is the reference's annotation time. nudge_s > 0 shows the
    reference later (its chords change nudge_s seconds after they would).
    """
    t = playback_s + lead_s
    frame = int(np.floor((t - (onset or 0.0)) * sr / hop))
    system_now, system_next = system_at(labels, frame)
    ref_now = ref_next = ""
    if reference is not None and onset is not None:
        ref_now, ref_next = reference.at(t - onset - nudge_s)
    return {"system": system_now, "system_next": system_next, "reference": ref_now, "reference_next": ref_next,
            "frame": frame, "ready": frame < len(labels)}


def held(now: dict, last: dict | None) -> dict:
    """`now` (from shown), with the system chord of `last` (the last shown state whose frame was labelled)
    while now's frame has started but is not labelled yet; before the first label there is nothing to keep."""
    if now["ready"] or now["frame"] < 0 or last is None:
        return now
    return {**now, "system": last["system"], "system_next": last["system_next"]}


def required_delay(point: str, lead_s: float, sr: int, hop: int, margin_frames: int = 1) -> float:
    """The smallest playback delay, s, for which every label is ready on time (one frame of margin)."""
    return (total_lookahead(point) + margin_frames) * hop / sr + lead_s


def load_system(cfg: dict, run: Path):
    """(model on CPU in eval mode, decoding prior) of a saved vocabulary-A run: model.pt and prior.json,
    the prior raised to cfg's decoder.prior_power."""
    import json

    import torch

    from chordtracker import model_v8, transitions_v8

    model = model_v8.ChordCRNN(len(CLASS_LABELS), bidirectional=False, dropout=cfg["train"]["dropout"])
    model.load_state_dict(torch.load(Path(run) / "model.pt", map_location="cpu"))
    prior = np.array(json.loads((Path(run) / "prior.json").read_text()))
    return model.eval(), transitions_v8.tempered_prior(prior, cfg["decoder"]["prior_power"])


def analysis_worker(blocks, results, cfg: dict, run_dir: str, Q: np.ndarray, rate: int, timing=None) -> None:
    """The analysis, in its own process so it never holds up the audio callbacks.

    blocks: ("reset", generation) or (generation, mono block at `rate`); results: (generation, {point:
    new labels}, onset time). Resamples to cfg's rate, finds the onset, runs Analysis. timing: a queue
    that gets (samples at `rate`, seconds) for every analysed block, or None.
    """
    import time as _time

    import soxr
    import torch

    torch.set_num_threads(1)
    model, prior = load_system(cfg, Path(run_dir))
    sr = cfg["audio"]["sr"]
    generation, analysis, onset, resampler, analysed, sent = None, None, None, None, 0, {}
    while True:
        message = blocks.get()
        if message is None:
            return
        if message[0] == "reset":
            generation = message[1]
            analysis, onset = Analysis(cfg, model, prior, Q), Onset()
            resampler = soxr.ResampleStream(rate, sr, 1, dtype="float32")
            analysed, sent = 0, {p: 0 for p in OPERATING_POINTS}
            continue
        if message[0] != generation or analysis is None:
            continue
        y = resampler.resample_chunk(message[1])
        onset.update(y, analysed / sr)
        analysed += len(y)
        if onset.time is None:
            continue                                # the song has not started: nothing to analyse yet
        start = _time.perf_counter()
        analysis.step(y)                            # from the onset block on: the tuning reads the song's 10 s
        if timing is not None:
            timing.put((len(message[1]), _time.perf_counter() - start))
        new = {p: analysis.labels[p][sent[p]:] for p in OPERATING_POINTS}
        if any(new.values()) or onset.time is not None:
            sent = {p: len(analysis.labels[p]) for p in OPERATING_POINTS}
            results.put((generation, new, onset.time))
