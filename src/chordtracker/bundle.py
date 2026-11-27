"""Component 42: a frozen system as a bundle, for a port of the live chain (the standalone macOS app).

The live chain (play_along.Analysis) runs in Python on numpy, librosa and torch. A port in another language
cannot import it, so the bundle holds what the port needs and the numbers it must reproduce
(reports/bundle.md describes every stage as a port computes it):

    system.json             the settings of every stage, the labels and their names, the operating points
                            with their delays, and the files below
    decoder/prior.f64       (n,) what the evidence divides by (online_hmm.evidence)
    decoder/lag<L>.f64      (n, n) the transition matrix A of each fixed-lag Viterbi point
    ChordStep.mlpackage     the network, one frame per call (FrameStep; scripts/export_bundle.py converts it)
    golden/<case>/          one recording through the live chain, every stage kept (capture), described by
                            its manifest.json

Arrays are raw little-endian files in row-major order, each described by {"file", "dtype", "shape"}:

    write_array(folder, name, array, dtype) -> its description (array_description); read_array back
    FrameStep(model)               the causal CRNN's step: (window, GRU state) -> (posteriors, new state)
    describe(...)                  system.json and the decoder arrays of a loaded chain
    capture(y, ...)                a recording through the live chain, with every stage's output
    tuning_pitches(y, sr)          the frequencies librosa.estimate_tuning makes its histogram of
    replay(predict, logcqt, net)   the posteriors of a step function fed the golden columns, as a port feeds it
    check(folder, predict)         every golden case through the bundle's own network and decoder arrays

Labels are decode indices (N last). Frame f is the 2,048-sample block at f * hop / sr of the recording.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

from chordtracker import demo, features_v7_1, model_v6, online_hmm, transitions
from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS, PATCH_FRAMES
from chordtracker.play_along import INPUT_LOOKAHEAD, SILENCE_DBFS, Analysis, chord_name, total_lookahead

FORMAT = 1                      # system.json's "format": raised when a file changes meaning
DTYPES = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "c64": "<c8"}
KERNEL_BINS = (0, 84, 167)      # the bins whose A440 kernel taps a golden case keeps (lowest, middle, highest)
LEAD_CHOICES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)    # the demo page's lead choices (demo/index.html), s


def array_description(name: str, array: np.ndarray, dtype: str) -> dict:
    """{"file": name.<dtype>, "dtype", "shape"}: how system.json and the manifests refer to an array."""
    return {"file": f"{name}.{dtype}", "dtype": dtype, "shape": list(np.shape(array))}


def write_array(folder: Path, name: str, array: np.ndarray, dtype: str) -> dict:
    """Write array as folder/name.<dtype> (raw little-endian, row-major); return its description."""
    description = array_description(name, array, dtype)
    path = Path(folder) / description["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.ascontiguousarray(array, dtype=DTYPES[dtype]).tobytes())
    return description


def read_array(folder: Path, description: dict) -> np.ndarray:
    """The array write_array described."""
    data = np.frombuffer((Path(folder) / description["file"]).read_bytes(), dtype=DTYPES[description["dtype"]])
    return data.reshape(description["shape"])


class FrameStep(nn.Module):
    """One frame of the causal CRNN: the 15-column window ending at frame t + a and the GRU state ->
    P(class | frame t) and the new state; LookaheadStreamingCRNN's step without the window bookkeeping.

        window  (1, 1, 144, 15)  rows CENTRE_ROW .. CENTRE_ROW + 144 of the log-CQT, oldest column first
        hidden  (1, 1, 64)       zeros before the first frame
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, window: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.model.gru(self.model.frame_embeddings(window)[None], hidden)
        return torch.softmax(self.model.classifier(out)[0, 0], dim=0), hidden


def point_file(point: dict) -> str:
    """The file stem of a fixed-lag point's transition matrix: lag 4 -> "lag4"."""
    return f"lag{point['lag']}"


def describe(system: str, cfg: dict, run: str, prior: np.ndarray, Q: np.ndarray, labels: list[str], points: dict,
             source: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """(system.json, {name: float64 array} to write with write_array) for a loaded chain (play_along.load_chain).

    points: the demo's points of the system (demo.MODELS), each with its total look-ahead and its delays as
    the page computes them (demo.point_info). source: where the bundle comes from (repository, commit).
    """
    a, q, comp = cfg["audio"], cfg["cqt"], cfg["compression"]
    sr, hop = a["sr"], q["hop_length"]
    arrays = {"decoder/prior": np.asarray(prior, dtype=float)}
    rows = []
    for info in demo.point_info(sr, hop, points):
        v = points[info["id"]]
        row = {**info, "lookaheadFrames": total_lookahead(info["id"]), "transitions": None}
        if v["p_self"] is not None:
            name = f"decoder/{point_file(v)}"
            arrays[name] = transitions.transition(Q, v["p_self"])
            row["transitions"] = array_description(name, arrays[name], "f64")
        rows.append(row)
    description = {
        "format": FORMAT,
        "system": system,
        "source": {**source, "config": f"configs/{cfg['version']}.yaml", "run": run},
        "audio": {"sampleRate": sr, "mono": True},
        "harmonic": {"nFft": features_v7_1.N_FFT, "hop": features_v7_1.STFT_HOP, "window": "hann, periodic",
                     "timeMedianFrames": a["harmonic_median_frames"], "frequencyMedianBins": features_v7_1.MEDIAN_FRAMES,
                     "margin": a["harmonic_margin"], "power": 2.0},
        "cqt": {"hop": hop, "fmin": q["fmin"], "fminHz": float(librosa.note_to_hz(q["fmin"])),
                "binsPerOctave": q["bins_per_octave"], "nBins": q["n_bins"], "filterScale": 1, "norm": 1,
                "window": "hann, periodic"},
        "tuning": {"seconds": q["tuning_seconds"], "audibleDbfs": SILENCE_DBFS, "estimateBinsPerOctave": 12,
                   "resolution": 0.01, "nFft": features_v7_1.N_FFT, "hop": features_v7_1.STFT_HOP,
                   "fminHz": 150.0, "fmaxHz": 4000.0, "threshold": 0.1},
        "compression": {"levelPercentile": comp["level_percentile"], "gamma": comp["gamma"], "levelFloor": 1e-10},
        "network": {"file": "ChordStep.mlpackage", "rows": [CENTRE_ROW, CENTRE_ROW + PATCH_BINS],
                    "windowFrames": PATCH_FRAMES, "inputLookaheadFrames": INPUT_LOOKAHEAD,
                    "hidden": model_v6.HIDDEN, "classes": len(labels),
                    "inputs": {"window": [1, 1, PATCH_BINS, PATCH_FRAMES], "hidden": [1, 1, model_v6.HIDDEN]},
                    "outputs": {"posteriors": [len(labels)], "hidden_out": [1, 1, model_v6.HIDDEN]}},
        "labels": [{"harte": label, "name": chord_name(label)} for label in labels],
        "decoder": {"eps": online_hmm.EPS, "prior": array_description("decoder/prior", prior, "f64"), "points": rows,
                    "defaultPoint": demo.DEFAULT_POINT, "framewisePoint": demo.FRAMEWISE_POINT},
        "display": {"leadSeconds": list(LEAD_CHOICES), "defaultLeadSeconds": 0.0, "delayOnByDefault": True,
                    "frameSeconds": round(hop / sr, 6)},
    }
    return description, arrays


@dataclass
class Recording(Analysis):
    """Analysis that keeps every log-CQT column and every posterior on the way."""
    logcqt: list = field(default_factory=list)
    posteriors: list = field(default_factory=list)

    def _column(self, column: np.ndarray) -> None:
        self.logcqt.append(column)
        super()._column(column)

    def _posterior(self, post: np.ndarray) -> None:
        self.posteriors.append(post)
        super()._posterior(post)


def tuning_pitches(y: np.ndarray, sr: int) -> np.ndarray:
    """The frequencies librosa.estimate_tuning (defaults: n_fft 2048, piptrack's) makes its histogram of: the
    peak pitches whose magnitude is at least the median of all peaks, in the (bins, frames) order of piptrack."""
    pitch, mag = librosa.piptrack(y=y, sr=sr, n_fft=features_v7_1.N_FFT)
    mask = pitch > 0
    threshold = np.median(mag[mask]) if mask.any() else 0.0
    return pitch[(mag >= threshold) & mask]


def bin_table(kernels, cfg: dict) -> dict:
    """Per CQT bin of a streaming_features.Kernels: frequency (Hz), librosa's (fractional) length, the trimmed
    kernel's taps and first tap (relative to the centre sample), the delay in frames and the scale (sqrt of
    the length)."""
    q = cfg["cqt"]
    hz = librosa.cqt_frequencies(q["n_bins"], fmin=librosa.note_to_hz(q["fmin"]), bins_per_octave=q["bins_per_octave"],
                                 tuning=kernels.tuning)
    return {"hz": [float(x) for x in hz], "length": [float(s) ** 2 for s in kernels.scale],
            "taps": [len(k) for k in kernels.kernels], "firstTap": [int(s) for s in kernels.starts],
            "delayFrames": [int(d) for d in kernels.delays], "scale": [float(s) for s in kernels.scale]}


def capture(y: np.ndarray, cfg: dict, model, prior: np.ndarray, Q: np.ndarray, points: dict, n_label: int,
            onset_dbfs: float | None, block: int) -> tuple[dict, dict[str, tuple[np.ndarray, str]]]:
    """(scalars, {name: (array, dtype)}) of y (mono, cfg's rate) through the live chain, blocks of `block`
    samples then finish(), with the silence gate off, as the demo runs a network with its own N:

        input (n,) f32, harmonic (n,) f32 the harmonic part, levels (T,) f64 dBFS of every frame,
        tuningHarmonic (n_tuning,) f32 the harmonic part the tuning is estimated on, tuningPitches f32 the
        frequencies of its histogram, cqt (T, 168) f64 |CQT| before the level and the log, logcqt (T, 168) f32
        the columns the network reads, posteriors (T, n) f32 in decode order, labels/<lag> (T,) i32 per point,
        kernelA440Bin<k> c64 the trimmed A440 kernel of a few bins.

    The |CQT| is recomputed from the streamed harmonic part with each column's kernels (A440, then the
    tuned ones), and checked against the streamed columns once compressed.
    """
    a = Recording(cfg, model, prior, Q, points, None, n_label, onset_dbfs)
    for i in range(0, len(y), block):
        a.step(y[i:i + block])
    a.finish()
    f, sr, comp = a.features, cfg["audio"]["sr"], cfg["compression"]
    logcqt = np.stack(a.logcqt)
    T = len(logcqt)
    C = f.kernels.all_columns(f.y_h.view, T)
    first = None
    if f.tuned is not None:
        first = int(f.first_tuned())
        C[:, first:] = f.tuned.all_columns(f.y_h.view, T)[:, first:]
    level = features_v7_1.running_level(C, comp["level_percentile"])
    recomputed = np.log1p(comp["gamma"] * C / level[None, :]).T
    arrays = {"input": (y, "f32"), "harmonic": (f.y_h.view, "f32"), "levels": (np.array(f.levels), "f64"),
              "cqt": (C.T, "f64"), "logcqt": (logcqt, "f32"), "posteriors": (np.stack(a.posteriors), "f32")}
    for k in KERNEL_BINS:
        arrays[f"kernelA440Bin{k}"] = (f.kernels.kernels[k], "c64")
    for point, v in points.items():
        arrays[f"labels/{point_file(v)}"] = (np.array(a.labels[point]), "i32")
    tuning = {"startSample": f.start, "seconds": cfg["cqt"]["tuning_seconds"], "bins": None, "semitones": None,
              "firstTunedColumn": first}
    if f.tuned is not None:
        y_t = features_v7_1.causal_harmonic(f.raw.view[f.start:f.start + f.n_tuning], sr, cfg)
        arrays["tuningHarmonic"] = (y_t, "f32")
        arrays["tuningPitches"] = (tuning_pitches(y_t, sr), "f32")
        tuning.update(bins=float(f.tuned.tuning), semitones=float(f.tuned.tuning) * 12 / cfg["cqt"]["bins_per_octave"])
    scalars = {"samples": len(y), "frames": T, "block": block, "tuning": tuning,
               "cqtBins": {"a440": bin_table(f.kernels, cfg),
                           "tuned": bin_table(f.tuned, cfg) if f.tuned is not None else None},
               "streamedVsRecomputedLogcqt": float(np.abs(recomputed - logcqt).max())}
    return scalars, arrays


def write_case(folder: Path, name: str, scalars: dict, arrays: dict, about: dict) -> dict:
    """golden/<name>/: every array and manifest.json (about: where the recording comes from)."""
    out = Path(folder) / "golden" / name
    files = {key: write_array(out, key, array, dtype) for key, (array, dtype) in arrays.items()}
    manifest = {"case": name, **about, **scalars, "arrays": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def replay(predict, logcqt: np.ndarray, network: dict) -> np.ndarray:
    """(T, n) posteriors of predict(window, hidden) -> (posteriors, hidden), fed the columns as a port feeds
    them: window of zeros at first, one column in per frame, a step once more than inputLookaheadFrames columns
    have arrived, inputLookaheadFrames zero columns at the end."""
    lo, hi = network["rows"]
    a = network["inputLookaheadFrames"]
    window = np.zeros((1, 1, hi - lo, network["windowFrames"]), dtype=np.float32)
    hidden = np.zeros(network["inputs"]["hidden"], dtype=np.float32)
    out = []
    columns = list(logcqt) + [np.zeros(logcqt.shape[1], dtype=np.float32)] * a
    for seen, column in enumerate(columns, 1):
        window = np.concatenate([window[..., 1:], column[lo:hi].astype(np.float32)[None, None, :, None]], axis=3)
        if seen > a:
            post, hidden = predict(window, hidden)
            out.append(np.asarray(post, dtype=np.float32).ravel())
            hidden = np.asarray(hidden, dtype=np.float32).reshape(network["inputs"]["hidden"])
    return np.stack(out)


def decode(posteriors: np.ndarray, point: dict, folder: Path, prior: np.ndarray) -> np.ndarray:
    """(T,) labels of one point of system.json from (T, n) posteriors: argmax without transitions, else the
    fixed-lag Viterbi on the bundle's matrix (the streaming decoder's decisions, flush included)."""
    if point["transitions"] is None:
        return posteriors.argmax(axis=1)
    A = read_array(folder, point["transitions"])
    return online_hmm.fixed_lag_viterbi(posteriors.T, A, prior, [point["lagFrames"]])[point["lagFrames"]]


def check(folder: Path, predict) -> dict:
    """{case: {"posteriorsMaxAbsDiff", "labelsAgree": {point: share of frames}}}: every golden case's columns
    through predict (replay), decoded with the bundle's prior and matrices, against the golden posteriors and
    labels. Reads only the bundle."""
    folder = Path(folder)
    system = json.loads((folder / "system.json").read_text())
    prior = read_array(folder, system["decoder"]["prior"])
    out = {}
    for manifest_file in sorted((folder / "golden").glob("*/manifest.json")):
        case = manifest_file.parent
        files = json.loads(manifest_file.read_text())["arrays"]
        post = replay(predict, read_array(case, files["logcqt"]), system["network"])
        golden = read_array(case, files["posteriors"])
        agree = {}
        for point in system["decoder"]["points"]:
            want = read_array(case, files[f"labels/lag{point['lagFrames']}"])
            agree[point["id"]] = float(np.mean(decode(post, point, folder, prior) == want))
        out[case.name] = {"posteriorsMaxAbsDiff": float(np.abs(post - golden).max()), "labelsAgree": agree}
    return out
