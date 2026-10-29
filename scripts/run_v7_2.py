"""Run V7.2: train the causal CRNN with an input look-ahead on the F#0 or C2 grid, then decode on validation.

    python scripts/run_v7_2.py --split val --train --input-lookahead 2 --seed 0             # -> results/v7_2/f0_la2_seed0
    python scripts/run_v7_2.py --split val --train --input-lookahead 0 --cqt-from C2 --seed 1  # -> results/v7_2/c2_la0_seed1
    python scripts/run_v7_2.py --split test --freeze                                      # once: every frozen point
    python scripts/run_v7_2.py --split val --train --limit 2 --max-epochs 1 --out <dir>   # dry run

The F#0 grid at input look-ahead 0 is V7.1 (results/v7_1/median9_seed*), not retrained. Decoders are
V7's (frame-argmax, forward-filter, fixed-lag-viterbi at every lag, offline-frame-viterbi); the score
is aligned WCSR (with playback delayed by the total look-ahead, every label arrives with its frame) and
chord changes per minute. scripts/v7_2_curve.py turns all runs into the best WCSR per total look-ahead.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import acoustic_hmm, data_v5, data_v7_2, experiment, paths, train_v7_2

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config, evaluation_sets, change_prior)
import run_v7  # noqa: E402  (frame_paths, score, summarize, best_p_self, system names)

FMIN = {"F#0": "F#0", "C2": "F#1"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v7_2.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--train", action="store_true", help="train the CRNN (val only); else load the saved one")
    parser.add_argument("--input-lookahead", type=int, default=0)
    parser.add_argument("--cqt-from", choices=list(FMIN), default="F#0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze", action="store_true", help="required for --split test: the val choices are final")
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int, help="dry run: override train.max_epochs")
    parser.add_argument("--out", type=Path, help="run directory; default results/v7_2/<grid>_la<a>_seed<seed>")
    args = parser.parse_args(argv)
    if (args.split == "test") != args.freeze or (args.split == "test" and args.train):
        parser.error("--split test needs --freeze and never trains; --freeze is only for test")
    return args


def run_dir(cqt_from: str, lookahead: int, seed: int) -> Path:
    """Where a cell's run lives; the F#0 grid at look-ahead 0 is V7.1's."""
    if cqt_from == "F#0" and lookahead == 0:
        return paths.REPO_ROOT / "results" / "v7_1" / f"median9_seed{seed}"
    return paths.REPO_ROOT / "results" / "v7_2" / f"{'f0' if cqt_from == 'F#0' else 'c2'}_la{lookahead}_seed{seed}"


def cell_config(cfg: dict, cqt_from: str, lookahead: int, seed: int) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["cqt"]["fmin"] = FMIN[cqt_from]
    cfg["train"]["input_lookahead"] = lookahead
    cfg["train"]["seed"] = seed
    return cfg


def score_take(model, take, cfg, Q, prior, grids, dataset) -> list[dict]:
    """Per-song rows (aligned WCSR only) of one take for every system, p_self and lag."""
    song = data_v7_2.load(take, cfg)
    post = train_v7_2.song_posteriors(model, song, cfg["train"]["input_lookahead"])
    times = acoustic_hmm.frame_times(post.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    rows = []
    for system, p_self, lag, path in run_v7.frame_paths(post, Q, prior, grids):
        aligned, changes = run_v7.score(path, times, take)
        rows.append({"dataset": dataset, "system": system, "p_self": p_self, "lag": lag, "song_id": take.key,
                     "changes_per_min": changes, "scored_s": aligned.scored_s, "aligned_s": aligned.matched_s,
                     "on_time_s": None, "end_to_end_s": None})
    return rows


def decode(model, prior, cfg, split, grids, first) -> list[dict]:
    Q = run_v5.change_prior(cfg)
    per_song = []
    for dataset, takes in run_v5.evaluation_sets(split).items():
        for take in first(takes):
            per_song += score_take(model, take, cfg, Q, prior, grids, dataset)
        print(f"{dataset}: {len(first(takes))} takes", flush=True)
    return per_song


def main(argv=None) -> None:
    args = parse_args(argv)
    base = run_v5.load_config(args.config)
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    d = base["decoder"]

    if args.split == "test":
        for point in base["frozen"]["points"]:
            cfg = cell_config(base, point["cqt_from"], point["input_lookahead"], point["seed"])
            run = run_dir(point["cqt_from"], point["input_lookahead"], point["seed"])
            model = train_v7_2.load(run / "model.pt", False, cfg["train"]["dropout"])
            prior = np.array(json.loads((run / "prior.json").read_text()))
            system, lag, p_self = point["system"], point["lag"], point["p_self"]
            grids = {run_v7.FILTER: [], run_v7.OFFLINE: [], run_v7.FIXED_LAG: {}}
            if system == run_v7.FIXED_LAG:
                grids[run_v7.FIXED_LAG] = {p_self: [lag]}
            elif system == run_v7.FILTER:
                grids[run_v7.FILTER] = [p_self]
            per_song = [r for r in decode(model, prior, cfg, "test", grids, first)
                        if r["system"] == system and r["lag"] == lag and r["p_self"] == p_self]
            rows = run_v7.summarize(per_song)
            point["test"] = {r["dataset"]: r["aligned"] for r in rows}
            point["test_changes_per_min"] = {r["dataset"]: r.get("mean_changes_per_min") for r in rows}
            print(point, flush=True)
        out = paths.REPO_ROOT / "results" / "v7_2" / "test"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"split": "test", "points": base["frozen"]["points"]}, indent=2))
        return

    cfg = cell_config(base, args.cqt_from, args.input_lookahead, args.seed)
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    out = args.out or run_dir(args.cqt_from, args.input_lookahead, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    if args.train:
        train_songs = [data_v7_2.load(take, cfg)
                       for take in first(data_v5.rwc_takes("train")) + first(data_v5.guitarset_takes("train"))]
        val_songs = {name: [data_v7_2.load(take, cfg) for take in first(takes)]
                     for name, takes in run_v5.evaluation_sets("val").items()}
        print(f"train: {len(train_songs)} songs, CQT from {args.cqt_from}, input look-ahead {args.input_lookahead}, "
              f"seed {args.seed}", flush=True)
        model, history = train_v7_2.train(train_songs, val_songs, cfg)
        train_v7_2.save(model, out / "model.pt")
        prior = acoustic_hmm.training_prior(np.concatenate([s.targets for s in train_songs]))
        (out / "prior.json").write_text(json.dumps(prior.tolist()))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"best epoch {max(history, key=lambda h: h['val_mean'])['epoch']} of {len(history)}, "
              f"{sum(h['seconds'] for h in history) / 60:.0f} min", flush=True)
    model = train_v7_2.load(out / "model.pt", False, cfg["train"]["dropout"])
    prior = np.array(json.loads((out / "prior.json").read_text()))
    grids = {run_v7.FILTER: d["p_self_grid"], run_v7.OFFLINE: d["p_self_grid"],
             run_v7.FIXED_LAG: {p_self: d["lags"] for p_self in d["p_self_grid"]}}
    per_song = decode(model, prior, cfg, "val", grids, first)
    rows = run_v7.summarize(per_song)
    summary = {"version": cfg["version"], "split": "val", "config": cfg, "rows": rows,
               "best_p_self": run_v7.best_p_self(rows)}
    experiment.write_results(summary, per_song, out / "val")
    for r in rows:
        if r["dataset"] == "mean" and r["system"] == run_v7.ARGMAX:
            print(f"  network alone, aligned {100 * r['aligned']:.2f}%")


if __name__ == "__main__":
    main()
