"""Run V7.1: train the causal CRNN on the causal log-CQT, then decode and score it as V7 does.

    python scripts/run_v7_1.py --split val --train --median-frames 31          # first run -> results/v7_1/seed0
    python scripts/run_v7_1.py --split val --train --median-frames 31 --seed 1 # -> results/v7_1/seed1
    python scripts/run_v7_1.py --split val --train [--seed 1]                  # config (correction 1, median 9)
                                                                              # -> results/v7_1/median9_seed<seed>
    python scripts/run_v7_1.py --split test --freeze --seed 1                 # once, p_self from `frozen`
    python scripts/run_v7_1.py --split val --train --limit 2 --max-epochs 1 --out <dir>   # dry run

Training data, validation and test sets are V5's; the input is features_v7_1's causal log-CQT (the
first run fills data/processed/v7_1_logcqt/). The decoders and the three WCSRs (aligned, on-time,
end-to-end) are run_v7's, without beat-viterbi-v6 (its beat tracker reads the whole song); the
input's look-ahead is front_end_lookahead_frames (1 frame).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import acoustic_hmm, data_v5, data_v7_1, experiment, paths, train_v7_1

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config, evaluation_sets, change_prior)
import run_v7  # noqa: E402  (frame_paths, score, shifted, summarize, best_p_self)

SYSTEMS = (run_v7.ARGMAX, run_v7.FILTER, run_v7.FIXED_LAG, run_v7.OFFLINE)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v7_1.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--train", action="store_true", help="train the CRNN (val only); else load the saved one")
    parser.add_argument("--seed", type=int, help="override train.seed")
    parser.add_argument("--median-frames", type=int,
                        help="override audio.harmonic_median_frames; results go to seed<seed> (31) or median<n>_seed<seed>")
    parser.add_argument("--freeze", action="store_true", help="required for --split test: the val choices are final")
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int, help="dry run: override train.max_epochs")
    parser.add_argument("--out", type=Path, help="run directory; default results/v7_1/seed<seed>")
    args = parser.parse_args(argv)
    if (args.split == "test") != args.freeze or (args.split == "test" and args.train):
        parser.error("--split test needs --freeze and never trains; --freeze is only for test")
    return args


def score_take(model, take, cfg, Q, prior, grids, dataset) -> list[dict]:
    """Per-song rows of one take for every system, p_self and lag (run_v7.score_take without beats)."""
    song = data_v7_1.load(take, cfg)
    post = train_v7_1.song_posteriors(model, song)
    times = acoustic_hmm.frame_times(post.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    front = cfg["front_end_lookahead_frames"]
    rows = []
    for system, p_self, lag, path in run_v7.frame_paths(post, Q, prior, grids):
        aligned, changes = run_v7.score(path, times, take)
        row = {"dataset": dataset, "system": system, "p_self": p_self, "lag": lag, "song_id": take.key,
               "changes_per_min": changes, "scored_s": aligned.scored_s, "aligned_s": aligned.matched_s,
               "on_time_s": None, "end_to_end_s": None}
        if lag is not None:
            row["on_time_s"] = run_v7.score(run_v7.shifted(path, lag), times, take)[0].matched_s
            row["end_to_end_s"] = run_v7.score(run_v7.shifted(path, lag + front), times, take)[0].matched_s
        rows.append(row)
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = run_v5.load_config(args.config)
    t, d = cfg["train"], cfg["decoder"]
    if args.max_epochs:
        t["max_epochs"] = args.max_epochs
    if args.seed is not None:
        t["seed"] = args.seed
    if args.median_frames:
        cfg["audio"]["harmonic_median_frames"] = args.median_frames
    width = cfg["audio"].get("harmonic_median_frames", 31)
    prefix = "" if width == 31 else f"median{width}_"
    out = args.out or paths.REPO_ROOT / "results" / cfg["version"] / f"{prefix}seed{t['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)

    if args.train:
        train_takes = first(data_v5.rwc_takes("train")) + first(data_v5.guitarset_takes("train"))
        train_songs = [data_v7_1.load(take, cfg) for take in train_takes]
        val_songs = {name: [data_v7_1.load(take, cfg) for take in first(takes)]
                     for name, takes in run_v5.evaluation_sets("val").items()}
        n_frames = sum(int((s.targets >= 0).sum()) for s in train_songs)
        print(f"train: {len(train_songs)} songs, {n_frames} frames with a target, seed {t['seed']}", flush=True)
        model, history = train_v7_1.train(train_songs, val_songs, cfg)
        train_v7_1.save(model, out / "model.pt")
        prior = acoustic_hmm.training_prior(np.concatenate([s.targets for s in train_songs]))
        (out / "prior.json").write_text(json.dumps(prior.tolist()))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"best epoch {max(history, key=lambda h: h['val_mean'])['epoch']} of {len(history)}, "
              f"{sum(h['seconds'] for h in history) / 60:.0f} min -> {out / 'model.pt'}", flush=True)
    model = train_v7_1.load(out / "model.pt", False, t["dropout"])
    prior = np.array(json.loads((out / "prior.json").read_text()))

    if args.split == "val":
        grids = {run_v7.FILTER: d["p_self_grid"], run_v7.OFFLINE: d["p_self_grid"],
                 run_v7.FIXED_LAG: {p_self: d["lags"] for p_self in d["p_self_grid"]}}
    else:
        frozen = cfg["frozen"]["p_self"]
        grids = {system: [frozen[system]] for system in (run_v7.FILTER, run_v7.OFFLINE)}
        grids[run_v7.FIXED_LAG] = {}
        for lag, p_self in frozen[run_v7.FIXED_LAG].items():
            grids[run_v7.FIXED_LAG].setdefault(p_self, []).append(int(lag))
    Q = run_v5.change_prior(cfg)
    per_song = []
    for dataset, takes in run_v5.evaluation_sets(args.split).items():
        for take in first(takes):
            per_song += score_take(model, take, cfg, Q, prior, grids, dataset)
        print(f"{dataset}: {len(first(takes))} takes", flush=True)
    rows = run_v7.summarize(per_song)
    summary = {"version": cfg["version"], "split": args.split, "config": cfg, "rows": rows}
    if args.split == "val":
        summary["best_p_self"] = run_v7.best_p_self(rows)
    experiment.write_results(summary, per_song, out / args.split)

    pct = lambda x: "   n/a" if x is None else f"{100 * x:6.2f}"
    for r in rows:
        best = summary.get("best_p_self", {}).get(f"{r['system']} lag {r['lag']}", {}).get("p_self")
        if r["dataset"] == "mean" and (args.split == "test" or r["p_self"] is None or best == r["p_self"]):
            print(f"  {r['system']:>21}  p_self {r['p_self']}  lag {r['lag']}  aligned {pct(r['aligned'])}  "
                  f"on-time {pct(r['on_time'])}  end-to-end {pct(r['end_to_end'])}")


if __name__ == "__main__":
    main()
