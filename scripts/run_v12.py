"""V12 on validation: decoder-side checks on the frozen V11 student (vocabulary A).

    python scripts/run_v12.py offsets       # per-quality log offsets (V10 step 2's method) on the V11 pool student

offsets: scripts/v10_offsets.py's fitting and both guards on V11's pool students, seed 0 and seed 1
(results/v11/pool_kd1_seed{0,1}, their val_posteriors.npz): (a) fit on one seed, judge on the other;
(b) on seed 1, fit on two val sets and judge on the third. The share table is at the frozen lag-8 p_self
(configs/v11.yaml). Written to results/v12/offsets/. Validation only; test is never loaded.
"""
import argparse
import sys
from pathlib import Path

import yaml

from chordtracker import paths

sys.path.insert(0, str(Path(__file__).parent))
import run_v10  # noqa: E402
import v10_offsets  # noqa: E402

SEEDS = {s: paths.REPO_ROOT / "results" / "v11" / f"pool_kd1_seed{s}" for s in (0, 1)}
CONFIG = paths.REPO_ROOT / "configs" / "v11.yaml"


def frozen_p_self(lag: int, config: Path = CONFIG) -> float:
    """The p_self of the frozen fixed-lag point at this lag."""
    points = yaml.safe_load(config.read_text())["frozen"]["points"]
    return next(p["p_self"] for p in points if p["lag"] == lag and p["system"] == "fixed-lag-viterbi")


def write(name: str, text: str, data) -> None:
    run_v10.write(name, text, data, version="v12")


def step_offsets() -> None:
    v10_offsets.run(run_v10.config(), run_v10.load_songs(), seeds=SEEDS, name="offsets", write=write,
                    share_p_self=frozen_p_self(8), ensembles=False, base="V11")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["offsets"])
    args = parser.parse_args(argv)
    if args.step == "offsets":
        step_offsets()


if __name__ == "__main__":
    main()
