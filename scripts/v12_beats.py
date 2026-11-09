"""Beats in the fixed-lag decoder, decode only: an upper bound on RWC-P and GuitarSet validation.

    python scripts/v12_beats.py      # -> results/v12/beats/{table.md, results.json}

The network and its saved validation posteriors are V11's frozen student (seed 1). The decoder is
the frozen lag-8 fixed-lag Viterbi, except that its self-transition probability depends on the frame:
p_on on frames within `width` frames of a beat, p_off elsewhere, so a chord change costs less near a
beat (beat_fixed_lag_viterbi). Beats come from:
    annotated   RWC-P's AIST beat annotations and GuitarSet's beat_position: the oracle
    downbeats   the annotated beats at position 1 of the bar only
    librosa     librosa.beat.beat_track on the full audio: an estimated stand-in, still offline
Controls: the frozen lag 8 (p_self 0.98) and the best constant p_self over a wider grid, so a gain
from beats is not just a gain from more smoothing. Settings are chosen on the mean sevenths of the
two sets, and also on one set and judged on the other. ChoCo is not here: its beats are not in data/.
"""
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import librosa
import numpy as np

import run_v10 as r
from chordtracker import frame_score, online_hmm, paths, transitions, transitions_v8

RUN = paths.REPO_ROOT / "results" / "v11" / "pool_kd1_seed1"
OUT = paths.REPO_ROOT / "results" / "v12" / "beats"
RWC_BEATS = paths.DATA_DIR / "rwc" / "rwc-annotations" / "01_annotations_preprocessed" / "beats" / "RWC-P"
GUITARSET = paths.DATA_DIR / "guitarset" / "annotation"
SETS = ("rwc", "guitarset")
LAG = 8
P_CONST = [0.95, 0.98, 0.99, 0.995, 0.998]
P_OFF = [0.98, 0.99, 0.995, 0.998]
P_ON = [0.3, 0.5, 0.7, 0.9, 0.95]
WIDTHS = [0, 1]


def beat_fixed_lag_viterbi(posteriors: np.ndarray, A_on: np.ndarray, A_off: np.ndarray, prior: np.ndarray,
                           on: np.ndarray, lag: int) -> np.ndarray:
    """online_hmm.fixed_lag_viterbi at one lag, with the move into frame t through A_on where on[t], else A_off.

    The decision for frame t still reads frames up to t + lag only, so `on` must be known by then.
    With A_on equal to A_off it is fixed_lag_viterbi.
    """
    log_b = online_hmm.evidence(posteriors, prior)
    logs = (np.log(A_off + online_hmm.EPS), np.log(A_on + online_hmm.EPS))
    n, T = log_b.shape
    delta = np.empty((T, n))
    psi = np.zeros((T, n), dtype=int)
    delta[0] = log_b[:, 0] - np.log(n)
    for t in range(1, T):
        scores = delta[t - 1][:, None] + logs[int(on[t])]
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores[psi[t], np.arange(n)] + log_b[:, t]
        delta[t] -= delta[t].max()
    frames = np.arange(T)
    read_at = np.minimum(frames + lag, T - 1)
    state = delta.argmax(axis=1)[read_at]
    for k in range(min(lag, T - 1)):
        step_back = read_at - k > frames
        state = np.where(step_back, psi[np.maximum(read_at - k, 0), state], state)
    return state


def near(beat_times: np.ndarray, T: int, width: int, sr: int, hop: int) -> np.ndarray:
    """(T,) bool: frame t (centred at t * hop / sr) within `width` frames of a beat."""
    on = np.zeros(T, dtype=bool)
    for f in np.round(np.asarray(beat_times) * sr / hop).astype(int):
        on[max(f - width, 0):min(f + width + 1, T)] = True
    return on


def annotated_beats(key: str) -> tuple[np.ndarray, np.ndarray]:
    """(beat times, downbeat times) of a validation song from its annotation."""
    dataset, name = key.split("/")
    if dataset == "rwc":
        with open(RWC_BEATS / f"{name}.csv") as f:
            rows = [(float(t), int(float(b))) for t, b in csv.reader(f, delimiter=";") if t != "t"]
    else:
        jams = json.loads((GUITARSET / f"{name}.jams").read_text())
        beats = next(a for a in jams["annotations"] if a["namespace"] == "beat_position")["data"]
        rows = [(b["time"], b["value"]["position"]) for b in beats]
    times = np.array([t for t, _ in rows])
    return times, np.array([t for t, b in rows if b == 1])


def librosa_beats(audio, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(audio), sr=sr, mono=True)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, units="time")
    return beats


_W: dict = {}


def _init(posts, prior, Q, beats, n_frames):
    _W.update(posts=posts, prior=prior, Q=Q, beats=beats, n_frames=n_frames)


def _decode(job):
    key, source, width, p_on, p_off, sr, hop = job
    post = _W["posts"][key]
    A_off = transitions.transition(_W["Q"], p_off)
    if source == "none":
        return job, online_hmm.fixed_lag_viterbi(post, A_off, _W["prior"], [LAG])[LAG]
    on = near(_W["beats"][source][key], post.shape[1], width, sr, hop)
    return job, beat_fixed_lag_viterbi(post, transitions.transition(_W["Q"], p_on), A_off, _W["prior"], on, LAG)


def changes_per_min(path: np.ndarray, sr: int, hop: int) -> float:
    return float(np.count_nonzero(np.diff(path))) / (len(path) * hop / sr / 60)


def main() -> None:
    cfg = r.config()
    sr, hop = cfg["audio"]["sr"], cfg["cqt"]["hop_length"]
    songs = r.load_songs()
    posts = {k: p for k, p in r.load_posteriors(RUN).items() if songs[k]["set"] in SETS}
    raw_prior = r.prior_of(RUN)
    prior = transitions_v8.tempered_prior(raw_prior, cfg["decoder"]["prior_power"])
    Q = r.run_v8.decoder_parts(cfg, r.VOCAB, raw_prior)
    takes = {t.key: t for name in SETS for t in r.val_takes(cfg)[name]}
    beats = {"annotated": {}, "downbeats": {}, "librosa": {}}
    for k in posts:
        beats["annotated"][k], beats["downbeats"][k] = annotated_beats(k)
    with ProcessPoolExecutor(8) as pool:
        for k, b in zip(posts, pool.map(librosa_beats, [takes[k].audio for k in posts], [sr] * len(posts))):
            beats["librosa"][k] = b
    jobs = [(k, "none", 0, None, p, sr, hop) for k in posts for p in P_CONST]
    jobs += [(k, s, w, on, off, sr, hop) for k in posts for s in beats for w, on, off in product(WIDTHS, P_ON, P_OFF)]
    systems: dict[tuple, dict] = {}
    with ProcessPoolExecutor(8, initializer=_init, initargs=(posts, prior, Q, beats, None)) as pool:
        for job, path in pool.map(_decode, jobs, chunksize=8):
            systems.setdefault(job[1:5], {})[job[0]] = path
    rows = []
    for system, paths_ in systems.items():
        sc = r.pooled({k: {("lag 8", 0): p} for k, p in paths_.items()}, songs, sets=SETS)
        row = {"source": system[0], "width": system[1], "p_on": system[2], "p_off": system[3]}
        for d in (*SETS, "mean"):
            row[d] = {m: 100 * sc[(d, "lag 8", 0)][m] for m in r.METRICS}
            keys = [k for k in paths_ if d == "mean" or songs[k]["set"] == d]
            row[d]["changes_per_min"] = float(np.mean([changes_per_min(paths_[k], sr, hop) for k in keys]))
        rows.append(row)
    reference = {d: float(np.mean([changes_per_min(np.maximum(songs[k]["targets"], -1), sr, hop)
                                   for k in posts if d == "mean" or songs[k]["set"] == d])) for d in (*SETS, "mean")}
    write(rows, reference, beats, posts, sr, hop)


def best(rows, source, on="mean"):
    cand = [x for x in rows if x["source"] == source]
    return max(cand, key=lambda x: x[on]["sevenths"])


def fmt(row) -> str:
    cells = [f"{row[d]['majmin']:.2f} / {row[d]['sevenths']:.2f} ({row[d]['changes_per_min']:.0f})" for d in (*SETS, "mean")]
    w = "" if row["source"] == "none" else row["width"]
    p_on = "" if row["p_on"] is None else f"{row['p_on']:g}"
    return f"| {row['source']} | {w} | {p_on} | {row['p_off']:g} | " + " | ".join(cells) + " |"


def write(rows, reference, beats, posts, sr, hop) -> None:
    head = ["| beats | width | p_on | p_off | RWC-P | GuitarSet | mean |", "|---|---:|---:|---:|---:|---:|---:|"]
    frozen = next(x for x in rows if x["source"] == "none" and x["p_off"] == 0.98)
    lines = ["Lag-8 fixed-lag Viterbi on V11 seed 1's validation posteriors, RWC-P and GuitarSet.",
             "Cells: majmin / sevenths % (chord changes per minute). Reference changes per minute: "
             + ", ".join(f"{d} {v:.0f}" for d, v in reference.items()) + ".", "",
             "Chosen on the mean sevenths of the two sets:", "", *head, fmt(frozen) + " frozen",
             fmt(best(rows, "none")) + " best constant"]
    lines += [fmt(best(rows, s)) for s in ("annotated", "downbeats", "librosa")]
    lines += ["", "Chosen on one set, judged on the other (sevenths %, change against the best constant chosen "
              "the same way):", "", "| beats | chosen on | judged on | setting (width, p_on, p_off) | sevenths | "
              "best constant | change |", "|---|---|---|---|---:|---:|---:|"]
    for s in ("annotated", "downbeats", "librosa"):
        for a, b in (("rwc", "guitarset"), ("guitarset", "rwc")):
            x, c = best(rows, s, a), best(rows, "none", a)
            lines.append(f"| {s} | {a} | {b} | {x['width']}, {x['p_on']:g}, {x['p_off']:g} | {x[b]['sevenths']:.2f} | "
                         f"{c[b]['sevenths']:.2f} (p {c['p_off']:g}) | {x[b]['sevenths'] - c[b]['sevenths']:+.2f} |")
    per_min = {s: float(np.mean([len(beats[s][k]) / (posts[k].shape[1] * hop / sr / 60) for k in posts]))
               for s in beats}
    lines += ["", "Beats per minute, mean over the songs: " + ", ".join(f"{s} {v:.0f}" for s, v in per_min.items()) + "."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    (OUT / "results.json").write_text(json.dumps({"rows": rows, "reference_changes_per_min": reference}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
