"""V10 step 4: a no-chord gate after decoding, N wherever the frame's log-CQT level is below a threshold.

Run by `python scripts/run_v10.py gate`. Level of frame t: the mean of its cached log-CQT column, i.e.
of log(1 + 10 C / running level), so it is relative to the song's level so far and 0 in digital silence.
The gate (decode_v10.level_gate) replaces the decoded class by N on those frames; nothing else changes.

Two scorings, from the same frame_score tables:
    with N     mir_eval majmin / sevenths as V8 reports them: reference N is scored, and only an N output
               matches it (V8, which never outputs N, gets every N second wrong)
    N-excluded reference N left out; the gate can then only lose, through false N on chords
Per set: reference N seconds, N seconds found (reference N, output N), false N seconds (output N where
the reference is a chord the metric scores). GuitarSet has no reference N: only its false N matters.
Decoders: seed 1 (V8 frozen), network alone and lag 8 at p_self 0.98.
"""
import numpy as np

import run_v10 as r
from chordtracker import decode_v10, frame_score

THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]       # 0: no gate (V8)
N_LABEL = r.N_CLASSES                                           # column 60 of the tables
POINTS = [("network", None), ("lag 8", 0.98)]


def gate_scores(paths_by_song, songs, threshold, point):
    """Per set: N found / false N seconds, reference N seconds, and WCSR with N and N-excluded."""
    acc = {}
    for key, by in paths_by_song.items():
        s = songs[key]
        path = decode_v10.level_gate(by[point], s["level"], threshold, N_LABEL)
        a = acc.setdefault(s["set"], {"ref_n": 0.0, "found_n": 0.0, "false_n": 0.0,
                                      **{f"{m}{v}": [0.0, 0.0] for m in r.METRICS for v in ("", "_non")}})
        out_n = path == N_LABEL
        ref_n = s["majmin_M"][:, N_LABEL]                        # seconds of reference N in each frame's cell
        a["ref_n"] += s["ref_n_s"]
        a["found_n"] += float(ref_n[out_n].sum())
        a["false_n"] += float((s["majmin_S"][:, N_LABEL] - ref_n)[out_n].sum())
        for m in r.METRICS:
            for v in ("", "_non"):
                matched, scored = frame_score.score_path(s[f"{m}{v}_M"], s[f"{m}{v}_S"], path)
                a[f"{m}{v}"][0] += matched
                a[f"{m}{v}"][1] += scored
    for a in acc.values():
        for m in r.METRICS:
            for v in ("", "_non"):
                a[f"{m}{v}"] = 100 * a[f"{m}{v}"][0] / a[f"{m}{v}"][1]
    return acc


def run(cfg, songs):
    posts = r.load_posteriors(r.SEEDS[1])
    paths = r.decode_all(posts, cfg, r.prior_of(r.SEEDS[1]), 0.25, p_selfs=[0.98], lags=[8])
    data, lines = {}, []
    for point in POINTS:
        name = r.POINT_NAMES[point[0]]
        lines += [f"Seed 1, {name}" + (f", p_self {point[1]:g}" if point[1] else "") + ". Seconds; WCSR %.", "",
                  "| set | threshold | ref N s | N found s | false N s | majmin with N | sevenths with N "
                  "| majmin N-excl. | sevenths N-excl. |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for th in THRESHOLDS:
            sc = gate_scores(paths, songs, th, point)
            data[f"{point[0]}|{th}"] = sc
            for d in r.SETS:
                a = sc[d]
                lines.append(f"| {d} | {th:g} | {a['ref_n']:.0f} | {a['found_n']:.0f} | {a['false_n']:.0f} | "
                             f"{a['majmin']:.2f} | {a['sevenths']:.2f} | {a['majmin_non']:.2f} | {a['sevenths_non']:.2f} |")
        lines.append("")
    r.write("step4_gate", "\n".join(lines), data)
    print("checked: threshold 0 equals V8:", check_zero(data), flush=True)


def check_zero(data) -> bool:
    return all(np.isclose(v["found_n"], 0) and np.isclose(v["false_n"], 0)
               for key, sc in data.items() if key.endswith("|0.0") for v in sc.values())
