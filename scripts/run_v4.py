"""Run V4 (V2 beat features, learned emission, Billboard change prior) over one split of GuitarSet.

    python scripts/run_v4.py --split val                  # fit on train, save the model, sweep p_self
    python scripts/run_v4.py --split test --p-self 0.40   # once, with the saved model and the p_self chosen on val

The emission is fitted on the train split only and saved to results/v4/model.json; the test run reads
it and never refits. Results go to results/v4/<split>/ (summary.json, per_song.csv), as run_v2.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import yaml

from chordtracker import beats, evaluate, experiment, learned_emission as le, paths, transitions
from chordtracker.data.data import load_manifest, load_split

MANIFEST = paths.DATA_DIR / "processed" / "guitarset" / "manifest.csv"
SPLIT = paths.SPLITS_DIR / "guitarset.json"


def parse_args(argv=None) -> argparse.Namespace:
    """--config (default configs/v4.yaml), --split val|test, --p-self (required for test, not allowed for val)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v4.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--p-self", type=float, help="the p_self chosen on val; required with --split test")
    args = parser.parse_args(argv)
    if args.split == "test" and args.p_self is None:
        parser.error("--split test needs --p-self (choose it on val first)")
    if args.split == "val" and args.p_self is not None:
        parser.error("--split val sweeps the grid from the config; --p-self is only for test")
    return args


def load_config(path: Path) -> dict:
    """The V4 config on top of its base (v2.yaml): V4 sections replace the base's."""
    cfg = yaml.safe_load(path.read_text())
    base = yaml.safe_load((path.parent / cfg["base_config"]).read_text())
    return {**base, **cfg}


def change_prior(cfg: dict) -> np.ndarray:
    """(24, 24) Q from the Billboard counts (V3 code), N dropped and the rows renormalized."""
    t = cfg["transitions"]
    counts = np.array(json.loads((paths.REPO_ROOT / t["counts"]).read_text())["counts"], dtype=float)
    Q = transitions.change_matrix(counts, alpha=t["alpha"], tau=t["tau"])[:le.N_CLASSES, :le.N_CLASSES]
    return Q / Q.sum(axis=1, keepdims=True)


def sweep(takes: list[beats.BeatEvidence], Q: np.ndarray, prior: np.ndarray, grid: list[float]):
    """Viterbi with A = p_self I + (1 - p_self) Q and p_state = prior at every p_self: rows as beats.sweep_p_self."""
    summary_rows, per_song_rows = [], []
    for p_self in grid:
        A = transitions.transition(Q, p_self)
        scores, changes = [], []
        for evidence in takes:
            path = librosa.sequence.viterbi_discriminative(evidence.posteriors, A, p_state=prior)
            song_score, changes_per_minute = beats.score_path(path, le.LABELS, evidence)
            scores.append(song_score)
            changes.append(changes_per_minute)
            per_song_rows.append({"system": p_self, "song_id": evidence.song_id, "wcsr": song_score.wcsr,
                                  "changes_per_min": changes_per_minute, "total_s": song_score.total_s,
                                  "scored_s": song_score.scored_s, "matched_s": song_score.matched_s,
                                  "reduced_s": song_score.reduced_s, "excluded_s": song_score.excluded_s})
        total = evaluate.pooled(scores)
        summary_rows.append({"system": p_self, "wcsr": total.wcsr, "mean_changes_per_min": float(np.mean(changes)),
                             "scored_s": total.scored_s, "reduced_s": total.reduced_s,
                             "excluded_s": total.excluded_s, "n_takes": len(takes)})
    return summary_rows, per_song_rows


def main(argv=None) -> None:
    """Fit on train (val run) or load the saved model (test run), decode the split, write and print the results."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    split = load_split(SPLIT)
    records = load_manifest(MANIFEST)
    model_path = paths.REPO_ROOT / "results" / cfg["version"] / "model.json"

    if args.split == "val":
        train = [le.take_beats(r, cfg) for r in records if r.song_id in set(split["train"])]
        model = le.fit(*le.training_rows(train), C=cfg["emission"]["C"])
        le.save(model, model_path)
        print(f"fitted on {len(train)} train takes -> {model_path}")
    else:
        model = le.load(model_path)

    takes = [le.to_evidence(le.take_beats(r, cfg), model) for r in records if r.song_id in set(split[args.split])]
    print(f"{args.split}: {len(takes)} takes")
    prior = model["prior"] if cfg["emission"]["divide_by_prior"] else None
    grid = cfg["decoder"]["p_self_grid"] if args.split == "val" else [args.p_self]
    summary_rows, per_song_rows = sweep(takes, change_prior(cfg), prior, grid)
    summary = {"version": cfg["version"], "split": args.split, "config": cfg, "rows": summary_rows}
    if args.split == "val":
        summary["chosen_p_self"] = experiment.choose_p_self(summary_rows)
    experiment.write_results(summary, per_song_rows, paths.REPO_ROOT / "results" / cfg["version"] / args.split)

    for row in summary_rows:
        print(f"  {row['system']:>5}  WCSR {100 * row['wcsr']:6.2f}%   {row['mean_changes_per_min']:7.2f} changes/min")
    if "chosen_p_self" in summary:
        print(f"chosen p_self: {summary['chosen_p_self']}")


if __name__ == "__main__":
    main()
