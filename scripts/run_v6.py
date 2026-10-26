"""Run V6 (CRNN acoustic model on the log-CQT, with or without V5's beat HMM) on RWC-P and GuitarSet.

    python scripts/run_v6.py --split val --train                       # bidirectional, seed 0 -> results/v6/bidir_seed0
    python scripts/run_v6.py --split val --train --seed 1              # -> results/v6/bidir_seed1
    python scripts/run_v6.py --split val --train --causal              # -> results/v6/causal_seed0
    python scripts/run_v6.py --split val --train --batch-excerpts 2 --out results/v6/bidir_b2_seed0
    python scripts/run_v6.py --split val                               # reuse the saved model, decode again
    python scripts/run_v6.py --split test --freeze --decoder hmm --p-self 0.4    # once, after the freeze
    python scripts/run_v6.py --split val --train --limit 2 --max-epochs 1 --out <dir>   # dry run

Training data, val and test sets, beat edges, the Billboard change prior and the p_self sweep are
run_v5's, imported unchanged; only the acoustic model differs. Two decoders are scored on the same
beat posteriors (CRNN frame posteriors pooled into beats by acoustic_hmm.frames_to_beats):

    argmax   the network alone: the most likely chord of every beat, no HMM
    hmm      V5's Viterbi: posteriors divided by the training prior, A = p_self I + (1 - p_self) Q

On val, the argmax row and every p_self of the grid are scored; summary.json keeps both. The test
split needs --freeze and one decoder (and its p_self), chosen on val; it is run once.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import acoustic_hmm, beats, data_v5, evaluate, experiment, paths, train_v6
from chordtracker.learned_emission import LABELS        # C:maj ... B:maj, C:min ... B:min, as data_v5.target

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config, evaluation_sets, beat_edges, sweep, change_prior)

ARGMAX = "argmax"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v6.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--train", action="store_true", help="train the CRNN (val only); else load the saved one")
    parser.add_argument("--seed", type=int, help="override train.seed")
    parser.add_argument("--causal", action="store_true", help="a forward-only GRU instead of a bidirectional one")
    parser.add_argument("--batch-excerpts", type=int, help="override train.batch_excerpts (excerpts per Adam step)")
    parser.add_argument("--freeze", action="store_true", help="required for --split test: the val choices are final")
    parser.add_argument("--decoder", choices=["hmm", ARGMAX], help="test only: the decoder chosen on val")
    parser.add_argument("--p-self", type=float, help="test only, with --decoder hmm: the p_self chosen on val")
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int, help="dry run: override train.max_epochs")
    parser.add_argument("--out", type=Path, help="run directory; default results/v6/<bidir|causal>_seed<seed>")
    args = parser.parse_args(argv)
    if args.split == "test":
        if not args.freeze or args.train or args.decoder is None:
            parser.error("--split test needs --freeze and --decoder, and never trains")
        if (args.decoder == "hmm") != (args.p_self is not None):
            parser.error("--p-self goes with --decoder hmm, and only with it")
    elif args.decoder or args.p_self is not None:
        parser.error("--split val scores the argmax row and the whole p_self grid; --decoder/--p-self are for test")
    return args


def evidence(model, take: data_v5.Take, cfg: dict) -> beats.BeatEvidence:
    """CRNN frame posteriors of one take (the whole song as one excerpt), pooled into its V1 beats."""
    song = data_v5.load(take, cfg)
    frame_post = train_v6.song_posteriors(model, song)
    times = acoustic_hmm.frame_times(frame_post.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    edges, duration = run_v5.beat_edges(take, cfg)
    return beats.BeatEvidence(take.key, acoustic_hmm.frames_to_beats(frame_post, times, edges), edges,
                              take.annotation, duration)


def argmax_rows(dataset: str, takes: list[beats.BeatEvidence]) -> tuple[list[dict], list[dict]]:
    """The network alone: the most likely chord of every beat. One summary row and per-song rows, as run_v5.sweep."""
    scores, changes, per_song_rows = [], [], []
    for ev in takes:
        song_score, changes_per_minute = beats.score_path(ev.posteriors.argmax(axis=0), LABELS, ev)
        scores.append(song_score)
        changes.append(changes_per_minute)
        per_song_rows.append({"dataset": dataset, "system": ARGMAX, "song_id": ev.song_id,
                              "wcsr": song_score.wcsr, "changes_per_min": changes_per_minute,
                              "total_s": song_score.total_s, "scored_s": song_score.scored_s,
                              "matched_s": song_score.matched_s, "reduced_s": song_score.reduced_s,
                              "excluded_s": song_score.excluded_s})
    total = evaluate.pooled(scores)
    return [{"dataset": dataset, "system": ARGMAX, "wcsr": total.wcsr, "mean_changes_per_min": float(np.mean(changes)),
             "scored_s": total.scored_s, "reduced_s": total.reduced_s, "excluded_s": total.excluded_s,
             "n_takes": len(takes)}], per_song_rows


def mean_rows(rows: list[dict]) -> list[dict]:
    """One 'mean' row per system: the mean of the RWC-P and GuitarSet WCSRs (the sets are never pooled)."""
    systems = list(dict.fromkeys(r["system"] for r in rows))
    return [{"dataset": "mean", "system": s, "wcsr": float(np.mean([r["wcsr"] for r in rows if r["system"] == s]))}
            for s in systems]


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = run_v5.load_config(args.config)
    t = cfg["train"]
    if args.max_epochs:
        t["max_epochs"] = args.max_epochs
    if args.seed is not None:
        t["seed"] = args.seed
    if args.causal:
        t["bidirectional"] = False
    if args.batch_excerpts:
        t["batch_excerpts"] = args.batch_excerpts
    out = args.out or paths.REPO_ROOT / "results" / cfg["version"] / (
        f"{'bidir' if t['bidirectional'] else 'causal'}_seed{t['seed']}")
    out.mkdir(parents=True, exist_ok=True)
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)

    if args.train:
        train_takes = first(data_v5.rwc_takes("train")) + first(data_v5.guitarset_takes("train"))
        train_songs = [data_v5.load(take, cfg) for take in train_takes]
        val_songs = {name: [data_v5.load(take, cfg) for take in first(takes)]
                     for name, takes in run_v5.evaluation_sets("val").items()}
        n_frames = sum(int((s.targets >= 0).sum()) for s in train_songs)
        print(f"train: {len(train_songs)} songs, {n_frames} frames with a target, "
              f"{'bidirectional' if t['bidirectional'] else 'causal'}, seed {t['seed']}", flush=True)
        model, history = train_v6.train(train_songs, val_songs, cfg)
        train_v6.save(model, out / "model.pt")
        prior = acoustic_hmm.training_prior(np.concatenate([s.targets for s in train_songs]))
        (out / "prior.json").write_text(json.dumps(prior.tolist()))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"best epoch {max(history, key=lambda h: h['val_mean'])['epoch']} of {len(history)}, "
              f"{sum(h['seconds'] for h in history) / 60:.0f} min -> {out / 'model.pt'}", flush=True)
    model = train_v6.load(out / "model.pt", t["bidirectional"], t["dropout"])
    prior = np.array(json.loads((out / "prior.json").read_text()))

    Q = run_v5.change_prior(cfg)
    hmm_grid = cfg["decoder"]["p_self_grid"] if args.split == "val" else [args.p_self] if args.decoder == "hmm" else []
    rows, per_song_rows = [], []
    for dataset, takes in run_v5.evaluation_sets(args.split).items():
        evidences = [evidence(model, take, cfg) for take in first(takes)]
        if args.split == "val" or args.decoder == ARGMAX:
            summary, songs = argmax_rows(dataset, evidences)
            rows += summary
            per_song_rows += songs
        if hmm_grid:
            summary, songs = run_v5.sweep(dataset, evidences, Q, prior, hmm_grid)
            rows += summary
            per_song_rows += songs
    means = mean_rows(rows)
    summary = {"version": cfg["version"], "split": args.split, "config": cfg, "rows": rows + means}
    if args.split == "val":
        summary["chosen_p_self"] = experiment.choose_p_self([r for r in means if r["system"] != ARGMAX])
    experiment.write_results(summary, per_song_rows, out / args.split)

    for row in rows + means:
        system = row["system"] if row["system"] == ARGMAX else f"p_self {row['system']:.2f}"
        print(f"  {row['dataset']:>9}  {system:>11}  WCSR {100 * row['wcsr']:6.2f}%")
    if "chosen_p_self" in summary:
        print(f"chosen p_self (mean of the two val sets): {summary['chosen_p_self']}")


if __name__ == "__main__":
    main()
