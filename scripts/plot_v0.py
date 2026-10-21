"""Make and save the V0 report figures (the same as chordtracker.report.v0_figures in a notebook).

    python scripts/plot_v0.py --p-self 0.95 --take 00_SS3-84-Bb_comp --take 04_Funk2-119-G_comp

Every figure comes from the val split and is written to figures/v0/val/. Test is scored, never
inspected: a test take is refused. Nothing is chosen here: --p-self must be the value already
chosen on val. Also prints the report tables.
"""
import argparse
from pathlib import Path

import yaml

from chordtracker import report


def parse_args(argv=None) -> argparse.Namespace:
    """--config (default configs/v0.yaml), --p-self (required), --take (a val song_id, repeatable)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=report.CONFIG)
    parser.add_argument("--p-self", type=float, required=True, help="the p_self chosen on val")
    parser.add_argument("--take", action="append", required=True, help="val song_id for a one-take figure")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Build the figures, save them, print where they are and the results tables."""
    args = parse_args(argv)
    version = yaml.safe_load(args.config.read_text())["version"]
    figures = report.v0_figures(args.p_self, args.take, args.config)
    print(f"figures in {report.save_figures(figures, version)}\n")
    print(report.results_tables(version))


if __name__ == "__main__":
    main()
