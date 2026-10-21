"""Run V1 (beat-synchronous chroma) over one split of GuitarSet: framewise vs Viterbi per beat.

    python scripts/run_v1.py --split val                  # sweep cfg["decoder"]["p_self_grid"], report the best
    python scripts/run_v1.py --split test --p-self 0.05   # once, with the p_self chosen on val

Same protocol and outputs as scripts/run_v0.py: results go to results/<version>/<split>/
(summary.json, per_song.csv), and the test split refuses to run without --p-self.
"""
import argparse
from pathlib import Path

import yaml

from chordtracker import beats, emission, experiment, paths
from chordtracker.data.data import load_manifest, load_split
from chordtracker.features import ChromaCQTExtractor

MANIFEST = paths.DATA_DIR / "processed" / "guitarset" / "manifest.csv"
SPLIT = paths.SPLITS_DIR / "guitarset.json"


def parse_args(argv=None) -> argparse.Namespace:
    """--config (default configs/v1.yaml), --split val|test, --p-self (required for test, not allowed for val)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v1.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--p-self", type=float, help="the p_self chosen on val; required with --split test")
    args = parser.parse_args(argv)
    if args.split == "test" and args.p_self is None:
        parser.error("--split test needs --p-self (choose it on val first)")
    if args.split == "val" and args.p_self is not None:
        parser.error("--split val sweeps the grid from the config; --p-self is only for test")
    return args


def main(argv=None) -> None:
    """Compute beat evidence once per take, decode with every system, score, write and print the results."""
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    extractor = ChromaCQTExtractor.from_config(cfg)
    model = emission.TemplateEmission.from_config(cfg)

    song_ids = set(load_split(SPLIT)[args.split])
    records = [record for record in load_manifest(MANIFEST) if record.song_id in song_ids]
    print(f"{args.split}: {len(records)} takes")
    takes = [beats.take_evidence(record, extractor, model, cfg) for record in records]

    grid = cfg["decoder"]["p_self_grid"] if args.split == "val" else [args.p_self]
    summary_rows, per_song_rows = beats.sweep_p_self(takes, model.labels, grid)
    summary = {"version": cfg["version"], "split": args.split, "config": cfg, "rows": summary_rows}
    if args.split == "val":
        summary["chosen_p_self"] = experiment.choose_p_self(summary_rows)

    out_dir = paths.REPO_ROOT / "results" / cfg["version"] / args.split
    experiment.write_results(summary, per_song_rows, out_dir)

    for row in summary_rows:
        print(f"  {str(row['system']):>9}  WCSR {100 * row['wcsr']:6.2f}%   "
              f"{row['mean_changes_per_min']:7.2f} changes/min")
    print(f"  scored {summary_rows[0]['scored_s']:.1f} s, of which reduced {summary_rows[0]['reduced_s']:.1f} s; "
          f"excluded {summary_rows[0]['excluded_s']:.1f} s")
    if "chosen_p_self" in summary:
        print(f"chosen p_self on val: {summary['chosen_p_self']}")
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
