"""Run V5.1 (V5 with an N class) on RWC-P and GuitarSet.

    python scripts/run_v5_1.py --split val --train          # train once, save the model, sweep p_self on val
    python scripts/run_v5_1.py --split val                  # reuse the saved model, sweep p_self again
    python scripts/run_v5_1.py --split test --p-self 0.4    # once, with the saved model and the p_self chosen on val
    python scripts/run_v5_1.py --split val --train --limit 2 --max-epochs 1 --out <dir>   # dry run

Same data, splits and log-CQT cache as run_v5; results go to results/v5.1/. On top of V5's rows,
every summary and per-song row carries the N split of evaluate's scored time (acoustic_hmm_v5_1.n_seconds):
ref_n, found, missed and false_n seconds. The data and beat helpers are run_v5's, imported unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import acoustic_hmm, acoustic_hmm_v5_1, beats, data_v5, data_v5_1, evaluate, experiment, paths, train_v5, train_v5_1

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config, evaluation_sets, beat_edges: unchanged for V5.1)

N_KEYS = ["ref_n", "found", "missed", "false_n"]


def change_prior(cfg: dict) -> np.ndarray:
    """(25, 25) Q from the Billboard counts, N row and column kept (acoustic_hmm_v5_1.change_prior)."""
    t = cfg["transitions"]
    counts = np.array(json.loads((paths.REPO_ROOT / t["counts"]).read_text())["counts"], dtype=float)
    return acoustic_hmm_v5_1.change_prior(counts, alpha=t["alpha"], tau=t["tau"])


def evidence(model, take: data_v5.Take, cfg: dict) -> beats.BeatEvidence:
    """25-class frame posteriors of one take, pooled into its beats (as run_v5.evidence)."""
    song = data_v5_1.load(take, cfg)
    frame_post = train_v5.song_posteriors(model, song)
    times = acoustic_hmm.frame_times(frame_post.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    edges, duration = run_v5.beat_edges(take, cfg)
    return beats.BeatEvidence(take.key, acoustic_hmm.frames_to_beats(frame_post, times, edges), edges,
                              take.annotation, duration)


def sweep(dataset: str, takes: list[beats.BeatEvidence], Q: np.ndarray, prior: np.ndarray, grid: list[float]):
    """Decode every take at every p_self; run_v5.sweep's rows plus the N seconds."""
    summary_rows, per_song_rows = [], []
    for p_self in grid:
        scores, changes, n_totals = [], [], dict.fromkeys(N_KEYS, 0.0)
        for ev in takes:
            path = acoustic_hmm.decode(ev.posteriors, Q, p_self, prior)
            estimate = beats.segments_to_annotation(path, acoustic_hmm_v5_1.LABELS, ev.edges)
            song_score = evaluate.score(ev.reference, estimate)
            changes_per_minute = evaluate.changes_per_minute(estimate, ev.duration)
            n = acoustic_hmm_v5_1.n_seconds(ev.reference, estimate)
            for key in N_KEYS:
                n_totals[key] += n[key]
            scores.append(song_score)
            changes.append(changes_per_minute)
            per_song_rows.append({"dataset": dataset, "system": p_self, "song_id": ev.song_id,
                                  "wcsr": song_score.wcsr, "changes_per_min": changes_per_minute,
                                  "total_s": song_score.total_s, "scored_s": song_score.scored_s,
                                  "matched_s": song_score.matched_s, "reduced_s": song_score.reduced_s,
                                  "excluded_s": song_score.excluded_s, **{f"n_{k}_s": v for k, v in n.items()}})
        total = evaluate.pooled(scores)
        summary_rows.append({"dataset": dataset, "system": p_self, "wcsr": total.wcsr,
                             "mean_changes_per_min": float(np.mean(changes)), "scored_s": total.scored_s,
                             "reduced_s": total.reduced_s, "excluded_s": total.excluded_s, "n_takes": len(takes),
                             **{f"n_{k}_s": v for k, v in n_totals.items()}})
    return summary_rows, per_song_rows


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in argv:
        argv += ["--config", str(paths.REPO_ROOT / "configs" / "v5_1.yaml")]
    args = run_v5.parse_args(argv)
    cfg = run_v5.load_config(args.config)
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    out = args.out or paths.REPO_ROOT / "results" / cfg["version"]
    out.mkdir(parents=True, exist_ok=True)
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)

    if args.train:
        train_takes = first(data_v5.rwc_takes("train")) + first(data_v5.guitarset_takes("train"))
        train_songs = [data_v5_1.load(t, cfg) for t in train_takes]
        val_songs = {name: [data_v5_1.load(t, cfg) for t in first(takes)]
                     for name, takes in run_v5.evaluation_sets("val").items()}
        targets = np.concatenate([s.targets for s in train_songs])
        print(f"train: {len(train_songs)} songs, {int((targets >= 0).sum())} frames with a target, "
              f"{int((targets == data_v5_1.N_CLASS).sum())} of them N", flush=True)
        model, history = train_v5_1.train(train_songs, val_songs, cfg)
        train_v5.save(model, out / "model.pt")
        prior = acoustic_hmm_v5_1.training_prior(targets)
        (out / "prior.json").write_text(json.dumps(prior.tolist()))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"prior: N {100 * prior[data_v5_1.N_CLASS]:.2f}%, each major {100 * prior[0]:.2f}%, "
              f"each minor {100 * prior[12]:.2f}%")
        print(f"best epoch {max(history, key=lambda h: h['val_mean'])['epoch']} of {len(history)} -> {out / 'model.pt'}")
    model = train_v5_1.load(out / "model.pt", cfg["train"]["dropout"])
    prior = np.array(json.loads((out / "prior.json").read_text()))

    Q = change_prior(cfg)
    grid = cfg["decoder"]["p_self_grid"] if args.split == "val" else [args.p_self]
    summary_rows, per_song_rows = [], []
    for dataset, takes in run_v5.evaluation_sets(args.split).items():
        evidences = [evidence(model, t, cfg) for t in first(takes)]
        rows, songs = sweep(dataset, evidences, Q, prior, grid)
        summary_rows += rows
        per_song_rows += songs
    mean_rows = [{"dataset": "mean", "system": p,
                  "wcsr": float(np.mean([r["wcsr"] for r in summary_rows if r["system"] == p]))} for p in grid]
    summary = {"version": cfg["version"], "split": args.split, "config": cfg, "rows": summary_rows + mean_rows}
    if args.split == "val":
        summary["chosen_p_self"] = experiment.choose_p_self(mean_rows)
    experiment.write_results(summary, per_song_rows, out / args.split)

    for row in summary_rows:
        print(f"  {row['dataset']:>9}  p_self {row['system']:.2f}  WCSR {100 * row['wcsr']:6.2f}%  "
              f"N found {row['n_found_s']:6.1f} / {row['n_ref_n_s']:6.1f} s  false N {row['n_false_n_s']:6.1f} s")
    for row in mean_rows:
        print(f"  {row['dataset']:>9}  p_self {row['system']:.2f}  WCSR {100 * row['wcsr']:6.2f}%")
    if "chosen_p_self" in summary:
        print(f"chosen p_self (mean of the two val sets): {summary['chosen_p_self']}")


if __name__ == "__main__":
    main()
