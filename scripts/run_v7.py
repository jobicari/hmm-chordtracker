"""Run V7 (decoders that only use past frames) on V6's causal CRNN, on RWC-P and GuitarSet.

    python scripts/run_v7.py --split val --run results/v6/causal_seed0          # -> results/v7/causal_seed0/val
    python scripts/run_v7.py --split val --run results/v6/causal_seed1
    python scripts/run_v7.py --split test --run results/v6/causal_seed1 --freeze  # once, p_self from the config
    python scripts/run_v7.py --split val --run results/v6/causal_seed0 --limit 2 --out <dir>   # dry run

Nothing is trained: the CRNN and its prior are read from the V6 run. Its frame posteriors (the whole
song in one pass, which streaming_crnn shows is what a frame-by-frame loop produces) are decoded by five systems:

    beat-viterbi-v6        V6's decoder: posteriors pooled per beat, Viterbi over the song (reference)
    frame-argmax           the most likely chord of every frame, the network alone (lag 0)
    forward-filter         online_hmm.forward_filter (lag 0), per frame
    fixed-lag-viterbi      online_hmm.fixed_lag_viterbi at every lag of the config, per frame
    offline-frame-viterbi  Viterbi over all frames of the song (reference)

Three WCSRs per system (mir_eval majmin, pooled per set, sets never pooled):

    aligned     each label compared with the reference at the frame it describes (t); the lag is
                reported, not penalized
    on_time     each label compared with the reference at the frame at which the decoder outputs
                it (t + L): the path shifted by L frames, "N" during the first L frames
    end_to_end  on_time, shifted further by the input's look-ahead (front_end_lookahead_frames): the
                earliest frame at which the label could exist with audio arriving as it is played

on_time and end_to_end are left out for the two whole-song references. On val every p_self of the grids is
scored; the test split needs --freeze and uses the p_self of every system (and of every lag of
fixed-lag-viterbi) from the config's `frozen` section, chosen on val.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

from chordtracker import acoustic_hmm, beats, data_v5, evaluate, experiment, online_hmm, paths, train_v6, transitions
from chordtracker.learned_emission import LABELS        # C:maj ... B:maj, C:min ... B:min

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config, evaluation_sets, change_prior, sweep)
import run_v6  # noqa: E402  (evidence: V6 posteriors pooled per beat)

BEAT, ARGMAX, FILTER, FIXED_LAG, OFFLINE = (
    "beat-viterbi-v6", "frame-argmax", "forward-filter", "fixed-lag-viterbi", "offline-frame-viterbi")
SYSTEMS = [BEAT, ARGMAX, FILTER, FIXED_LAG, OFFLINE]
NOTHING = len(LABELS)                                   # index of "N": no label output yet
KINDS = ("aligned", "on_time", "end_to_end")            # the three WCSRs (module docstring)
SHOWN = LABELS + ["N"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v7.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--run", type=Path, required=True, help="a causal V6 run: model.pt and prior.json")
    parser.add_argument("--freeze", action="store_true", help="required for --split test: the val choices are final")
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--out", type=Path, help="default results/v7/<name of the V6 run>")
    args = parser.parse_args(argv)
    if (args.split == "test") != args.freeze:
        parser.error("--freeze goes with --split test, and only with it")
    return args


def shifted(path: np.ndarray, frames: int) -> np.ndarray:
    """The path delayed by `frames`: frame t gets the chord of frame t - frames, "N" before that.

    Example: path [3, 3, 5, 5], frames 1 -> [N, 3, 3, 5].
    """
    out = np.full_like(path, NOTHING)
    if frames < len(path):
        out[frames:] = path[:len(path) - frames]
    return out


def score(path: np.ndarray, times: np.ndarray, take: data_v5.Take) -> tuple[evaluate.SongScore, float]:
    """(SongScore, changes per minute) of a frame path against the take's reference."""
    duration = float(take.annotation.intervals[-1, 1])
    estimate = evaluate.path_to_annotation(path, SHOWN, times, duration)
    return evaluate.score(take.annotation, estimate), evaluate.changes_per_minute(estimate, duration)


def frame_paths(post: np.ndarray, Q: np.ndarray, prior: np.ndarray, grids: dict):
    """Every frame-level (system, p_self, lag, label path) of one song. Lag None: whole song.

    grids[FIXED_LAG] maps each p_self to the lags decoded with it (all lags on val, one p_self per lag on test).
    """
    yield ARGMAX, None, 0, post.argmax(axis=0)
    for p_self in grids[FILTER]:
        yield FILTER, p_self, 0, online_hmm.forward_filter(post, transitions.transition(Q, p_self), prior)
    for p_self, lags in grids[FIXED_LAG].items():
        for lag, path in online_hmm.fixed_lag_viterbi(post, transitions.transition(Q, p_self), prior, lags).items():
            yield FIXED_LAG, p_self, lag, path
    for p_self in grids[OFFLINE]:
        A = transitions.transition(Q, p_self)
        yield OFFLINE, p_self, None, librosa.sequence.viterbi_discriminative(post, A, p_state=prior)


def score_take(model, take, cfg, Q, prior, grids, dataset) -> list[dict]:
    """Per-song rows of one take for every system, p_self and lag."""
    rows = []
    song = data_v5.load(take, cfg)
    post = train_v6.song_posteriors(model, song)
    times = acoustic_hmm.frame_times(post.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    front = cfg["front_end_lookahead_frames"]
    for system, p_self, lag, path in frame_paths(post, Q, prior, grids):
        aligned, changes = score(path, times, take)
        row = {"dataset": dataset, "system": system, "p_self": p_self, "lag": lag, "song_id": take.key,
               "changes_per_min": changes, "scored_s": aligned.scored_s, "aligned_s": aligned.matched_s,
               "on_time_s": None, "end_to_end_s": None}
        if lag is not None:
            row["on_time_s"] = score(shifted(path, lag), times, take)[0].matched_s
            row["end_to_end_s"] = score(shifted(path, lag + front), times, take)[0].matched_s
        rows.append(row)
    ev = run_v6.evidence(model, take, cfg)
    for p_self in grids[BEAT]:
        path = acoustic_hmm.decode(ev.posteriors, Q, p_self, prior)
        aligned, changes = beats.score_path(path, LABELS, ev)
        rows.append({"dataset": dataset, "system": BEAT, "p_self": p_self, "lag": None, "song_id": take.key,
                     "changes_per_min": changes, "scored_s": aligned.scored_s, "aligned_s": aligned.matched_s,
                     "on_time_s": None, "end_to_end_s": None})
    return rows


def summarize(per_song: list[dict]) -> list[dict]:
    """Pooled WCSR per (dataset, system, p_self, lag) and kind, then the mean of the two sets."""
    keys = list(dict.fromkeys((r["dataset"], r["system"], r["p_self"], r["lag"]) for r in per_song))
    rows = []
    for dataset, system, p_self, lag in keys:
        songs = [r for r in per_song if (r["dataset"], r["system"], r["p_self"], r["lag"]) == (dataset, system, p_self, lag)]
        scored = sum(r["scored_s"] for r in songs)
        row = {"dataset": dataset, "system": system, "p_self": p_self, "lag": lag, "n_takes": len(songs),
               "mean_changes_per_min": float(np.mean([r["changes_per_min"] for r in songs]))}
        for kind in KINDS:
            matched = [r[f"{kind}_s"] for r in songs]
            row[kind] = None if matched[0] is None else sum(matched) / scored
        rows.append(row)
    for system, p_self, lag in dict.fromkeys(k[1:] for k in keys):
        both = [r for r in rows if (r["system"], r["p_self"], r["lag"]) == (system, p_self, lag)]
        rows.append({"dataset": "mean", "system": system, "p_self": p_self, "lag": lag,
                     **{kind: None if both[0][kind] is None else float(np.mean([r[kind] for r in both]))
                        for kind in KINDS}})
    return rows


def best_p_self(rows: list[dict]) -> dict:
    """On val: for every (system, lag), the p_self with the best mean aligned WCSR (ties: the smaller)."""
    best = {}
    for r in rows:
        if r["dataset"] != "mean" or r["p_self"] is None:
            continue
        key = f"{r['system']} lag {r['lag']}"
        if key not in best or (r["aligned"], -r["p_self"]) > (best[key]["aligned"], -best[key]["p_self"]):
            best[key] = {"p_self": r["p_self"], "aligned": r["aligned"], "on_time": r["on_time"]}
    return best


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = run_v5.load_config(args.config)
    d = cfg["decoder"]
    if args.split == "val":
        grids = {BEAT: d["beat_p_self_grid"], FILTER: d["p_self_grid"], OFFLINE: d["p_self_grid"],
                 FIXED_LAG: {p_self: d["lags"] for p_self in d["p_self_grid"]}}
    else:
        frozen = cfg["frozen"]["p_self"]
        grids = {system: [frozen[system]] for system in (BEAT, FILTER, OFFLINE)}
        grids[FIXED_LAG] = {}
        for lag, p_self in frozen[FIXED_LAG].items():
            grids[FIXED_LAG].setdefault(p_self, []).append(int(lag))
    out = args.out or paths.REPO_ROOT / "results" / cfg["version"] / args.run.name
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)

    model = train_v6.load(args.run / "model.pt", cfg["train"]["bidirectional"], cfg["train"]["dropout"])
    prior = np.array(json.loads((args.run / "prior.json").read_text()))
    Q = run_v5.change_prior(cfg)
    per_song = []
    for dataset, takes in run_v5.evaluation_sets(args.split).items():
        for take in first(takes):
            per_song += score_take(model, take, cfg, Q, prior, grids, dataset)
        print(f"{dataset}: {len(first(takes))} takes", flush=True)
    rows = summarize(per_song)
    summary = {"version": cfg["version"], "split": args.split, "run": str(args.run), "config": cfg, "rows": rows}
    if args.split == "val":
        summary["best_p_self"] = best_p_self(rows)
    experiment.write_results(summary, per_song, out / args.split)

    pct = lambda x: "   n/a" if x is None else f"{100 * x:6.2f}"
    for r in rows:
        if r["dataset"] == "mean" and (args.split == "test" or r["system"] in (ARGMAX,) or
                                       summary["best_p_self"].get(f"{r['system']} lag {r['lag']}", {}).get("p_self") == r["p_self"]):
            print(f"  {r['system']:>21}  p_self {r['p_self']}  lag {r['lag']}  "
                  f"aligned {pct(r['aligned'])}  on-time {pct(r['on_time'])}  end-to-end {pct(r['end_to_end'])}")


if __name__ == "__main__":
    main()
