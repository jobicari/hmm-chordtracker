"""The listening prototype's logic, without audio or window (scripts/play_along.py runs it live).

A recording reaches the prototype as it plays (from a file read at real-time pace, or from another
application through a virtual audio cable). It is analysed block by block with the live chain and one of
two systems (SYSTEMS), and played back to the musician `delay` seconds late, so the label of every frame
is ready when that frame is heard:

    v11    the frozen V11 student (vocabulary A: maj, min, 7, maj7, min7, 60 classes; configs/v11.yaml
           `frozen`), with the silence gate for N (the default)
    v14    V14's network with an N class (61 classes, N last; results/v14/n_seed0, configs/v14.yaml) at
           its best decoder setting on validation (N weight x0.1, chord -> N rate x0.5): N comes from the
           network, the silence gate is off unless asked for. V14 did not pass its checks (reports/v14.md:
           it calls N on some chords); this is for listening to a learned N

This module holds what does not depend on sound devices:

    OPERATING_POINTS               V11's four frozen points (input look-ahead 2): the decoder lag and
                                   p_self of each, and so its total look-ahead
    V14_POINTS                     V14's points: the same lags, V8's p_self
    SYSTEMS, System                the two systems: config, run, points, silence gate on or off by default
    load_chain(system, cfg, run)   (model, decoding prior, Q) of a system's saved run
    V7_2_POINTS                    the demo website's three points (V7.2's network and p_self)
    CLASS_LABELS                   the Harte label of each of the 60 classes ('C:maj' .. 'B:min7')
    SILENCE_DBFS, SILENCE_FRAMES   the silence gate: -55 dBFS for 3 frames
    Analysis                       the live chain fed with blocks; per frame, the label of every point
                                   (N, index n_label, where the gate finds silence); finish() at the end
                                   of a recording
    silent_frames(levels, ...)     which frames the gate labels N, from every frame's dBFS
    Reference                      an Isophonics annotation, every chord reduced to vocabulary A (the
                                   sevenths mapping): the chord at a time and the next one
    reference_name(label)          "D:7(b9)" -> "D7", "C:maj6" -> "C", "B:dim" -> "(Bdim)" (not in the vocabulary)
    song_list(root)                every annotated song, by album then title, for the drop-down
    chord_name(label)              "E:min7" -> "Em7", "G:7" -> "G7", "A" -> "A", "N" -> "N.C."
    shown(...)                     what the window shows at a playback time: system chord now and next,
                                   reference chord now and next
    held(now, last)                the last labelled system chord, kept while the current frame is not
                                   labelled yet (the tuning step at 10 s can hold the analysis ~0.1 s)
    Onset                          the time the input first rises out of silence (the annotation's zero)

The decoders are V8's: fixed-lag Viterbi on log P(class | frame) - log(prior ** prior_power), with the
Billboard change matrix Q and the point's p_self on the diagonal.

Neither network has an N class, so silence would show a chord. The silence gate
labels a frame N when its own samples and those of the frames just before it are all below a level
(silent_frames: frame f is N once frames f - 2 .. f are below -55 dBFS, before the song counting as
silent). The decoders run underneath, so the chord returns with the first audible frame. Three frames
(0.28 s) keep the short rests of a performance: on the 60 GuitarSet validation takes a gate on single
frames at -50 dBFS would label 703 frames N in 35 takes, this one 117 frames in 4 takes (the pauses of
a player), and none in the demo's samples. Digital silence and quiet noise (-60 dBFS) are N. With the
gate the tuning seconds start at the first frame at or above the level (StreamingFeatures), so a quiet
intro no longer stands for the tuning of the whole recording.

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
SILENCE_DBFS = -55.0        # the gate's level: a frame below it is silent
SILENCE_FRAMES = 3          # frames in a row below it (0.28 s) before the gate labels N
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
V14_POINTS = {              # V14's points (configs/v14.yaml decoder.points, V8's frozen p_self)
    "0.28 s": {"lag": 0, "p_self": None},
    "0.46 s": {"lag": 2, "p_self": 0.5},
    "0.65 s": {"lag": 4, "p_self": 0.8},
    "1.02 s": {"lag": 8, "p_self": 0.98},
}
V14_N_WEIGHT = 0.1          # N's decoding evidence times this (reports/v14.md, best setting on validation)
V14_RATE_FACTOR = 0.5       # the counted chord -> N rate times this
N_LABEL = len(CLASS_LABELS)  # N in vocabulary A: V14's class 60, and the gate's label for the 60-class models
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


def silent_frames(levels: np.ndarray, threshold: float = SILENCE_DBFS, frames: int = SILENCE_FRAMES) -> np.ndarray:
    """(T,) bool: frame f is silent when frames f - frames + 1 .. f are all below threshold dBFS (frames
    before the first count as below). levels: the dBFS of every frame (streaming_features.frame_dbfs).

    Example, threshold -55, frames 3: levels [-80, -30, -60, -60, -60, -30] -> [T, F, F, F, T, F].
    """
    quiet, out, run = np.asarray(levels) < threshold, [], frames
    for q in quiet:
        run = run + 1 if q else 0
        out.append(run >= frames)
    return np.array(out, dtype=bool)


@dataclass
class Analysis:
    """The live chain on 22,050 Hz mono blocks; labels[point][f] = the class index of frame f, per point.

    prior: what the decoders divide by (transitions_v8.tempered_prior of the training prior); Q: the change
    matrix (zero diagonal) that each point's p_self completes; points: OPERATING_POINTS or V7_2_POINTS.
    silence_dbfs: the silence gate's level (None: no gate, the tuning seconds from time 0); a silent frame
    gets N at every point, index n_label (default len(prior): the class after the chords of a network
    without N; a network with an N class passes its own).
    """
    cfg: dict
    model: object
    prior: np.ndarray
    Q: np.ndarray
    points: dict = field(default_factory=lambda: OPERATING_POINTS)
    silence_dbfs: float | None = None
    n_label: int | None = None

    def __post_init__(self):
        self.labels = {p: [] for p in self.points}
        self.n_index = len(self.prior) if self.n_label is None else self.n_label
        self.features = StreamingFeatures(self.cfg, self.silence_dbfs)
        self.silent, self.run = [], SILENCE_FRAMES              # the gate per frame, and the quiet frames in a row
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
            for state in decoder.flush():
                self._append(point, state)

    def _column(self, column: np.ndarray) -> None:
        post = self.crnn.step(column[CENTRE_ROW:CENTRE_ROW + PATCH_BINS])
        if post is not None:
            self._posterior(post)

    def _posterior(self, post: np.ndarray) -> None:
        for point, v in self.points.items():
            if v["p_self"] is None:
                self._append(point, int(post.argmax()))
            else:
                state = self.decoders[point].step(online_hmm.evidence(post[:, None], self.prior)[:, 0])
                if state is not None:
                    self._append(point, state)

    def _append(self, point: str, state: int) -> None:
        """The label of the point's next frame: the decoder's state, or N where the gate finds silence."""
        f = len(self.labels[point])
        self.labels[point].append(self.n_index if self._is_silent(f) else int(state))

    def _is_silent(self, f: int) -> bool:
        """The gate of frame f (silent_frames), extended as the frames' levels become known. A frame's label
        always comes after its samples (its CQT column reads audio past the frame's end)."""
        if self.silence_dbfs is None:
            return False
        levels = self.features.levels
        while len(self.silent) <= f:
            self.run = self.run + 1 if levels[len(self.silent)] < self.silence_dbfs else 0
            self.silent.append(self.run >= SILENCE_FRAMES)
        return self.silent[f]


def class_name(index: int) -> str:
    """A label index as a musician reads it: a class's name, "N.C." for N (index len(CLASS_LABELS))."""
    return chord_name(CLASS_LABELS[index]) if index < len(CLASS_LABELS) else chord_name("N")


def system_at(labels: list[int], frame: int) -> tuple[str, str]:
    """(chord of frame, next different chord already decided) from a point's labels; "" if not known yet."""
    if frame < 0 or frame >= len(labels):
        return "", ""
    now = class_name(labels[frame])
    for later in labels[frame + 1:]:
        if class_name(later) != now:
            return now, class_name(later)
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


@dataclass(frozen=True)
class System:
    """A system the play-along can run: its config and saved run (relative to the repository), its
    operating points, and whether the silence gate is on unless told otherwise."""
    config: str
    run: str
    points: dict
    gate: bool


SYSTEMS = {
    "v11": System("configs/v11.yaml", "results/v11/pool_kd1_seed1", OPERATING_POINTS, gate=True),
    "v14": System("configs/v14.yaml", "results/v14/n_seed0", V14_POINTS, gate=False),
}


def billboard_counts(cfg: dict) -> np.ndarray:
    """The Billboard change counts the transition matrices are built from (cfg's transitions.counts)."""
    import json

    from chordtracker import paths

    return np.array(json.loads((paths.REPO_ROOT / cfg["transitions"]["counts"]).read_text())["counts"], dtype=float)


def load_chain(system: str, cfg: dict, run: Path):
    """(model on CPU in eval mode, decoding prior, Q) of a system's saved run.

    v11: load_system, and V8's Billboard matrix weighted by the training prior (run_v8.decoder_parts).
    v14: model_v14's 61-class network; the training prior (N last) raised to the prior power, N's entry
    divided by V14_N_WEIGHT (its evidence times the weight); Q with N's row and column at the counted chord
    -> N rate (n_counts.json) times V14_RATE_FACTOR (run_v14.specs).
    """
    import json

    from chordtracker import model_v14, transitions_v8

    run = Path(run)
    t = cfg["transitions"]
    prior = np.array(json.loads((run / "prior.json").read_text()))
    if system == "v11":
        model, decode_prior = load_system(cfg, run)
        Q = transitions_v8.change_matrix(billboard_counts(cfg), VOCAB, prior, t["alpha"], t["tau"])
        return model, decode_prior, Q
    if system != "v14":
        raise ValueError(f"unknown system {system!r}: one of {sorted(SYSTEMS)}")
    rate = json.loads((run / "n_counts.json").read_text())["chord_to_n_rate"] * V14_RATE_FACTOR
    Q = transitions_v8.change_matrix(billboard_counts(cfg), VOCAB, prior, t["alpha"], t["tau"], rate)
    decode_prior = transitions_v8.tempered_prior(prior, cfg["decoder"]["prior_power"])
    decode_prior[-1] /= V14_N_WEIGHT
    return model_v14.load(run, cfg).cpu(), decode_prior / decode_prior.sum(), Q


def analysis_worker(blocks, results, cfg: dict, run_dir: str, rate: int, timing=None, system: str = "v11",
                    silence_dbfs: float | None = SILENCE_DBFS) -> None:
    """The analysis, in its own process so it never holds up the audio callbacks.

    blocks: ("reset", generation) or (generation, mono block at `rate`); results: (generation, {point:
    new labels}, onset time). Resamples to cfg's rate, finds the onset, runs Analysis with the system's
    chain (load_chain) and the silence gate at silence_dbfs (None: off). timing: a queue that gets
    (samples at `rate`, seconds) for every analysed block, or None.
    """
    import time as _time

    import soxr
    import torch

    torch.set_num_threads(1)
    model, prior, Q = load_chain(system, cfg, Path(run_dir))
    points = SYSTEMS[system].points
    sr = cfg["audio"]["sr"]
    generation, analysis, onset, resampler, analysed, sent = None, None, None, None, 0, {}
    while True:
        message = blocks.get()
        if message is None:
            return
        if message[0] == "reset":
            generation = message[1]
            analysis = Analysis(cfg, model, prior, Q, points, silence_dbfs, n_label=N_LABEL)
            onset = Onset()
            resampler = soxr.ResampleStream(rate, sr, 1, dtype="float32")
            analysed, sent = 0, {p: 0 for p in points}
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
        new = {p: analysis.labels[p][sent[p]:] for p in points}
        if any(new.values()) or onset.time is not None:
            sent = {p: len(analysis.labels[p]) for p in points}
            results.put((generation, new, onset.time))
