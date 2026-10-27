"""V5's network alone on val: the argmax per beat of V5's CNN posteriors, with no HMM, for both V5 seeds.

    python scripts/v6_v5_argmax.py      # -> results/v6/v5_argmax.json

V6 asks whether a recurrent layer adds to the HMM or replaces it, so V6's argmax row needs V5's to
compare with: how much the HMM adds on top of a frame classifier with 15 frames of context. The two V5
models are the tagged V5 (seed 0, results/v5/model.pt) and its seed-1 retrain from V5.2
(results/v5.2/step1/v5_seed1/model.pt). Beat evidence is run_v5.evidence, scoring run_v6.argmax_rows.
Val only.
"""
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import paths, train_v5

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402
import run_v6  # noqa: E402

MODELS = {"v5_seed0": paths.REPO_ROOT / "results" / "v5" / "model.pt",
          "v5_seed1": paths.REPO_ROOT / "results" / "v5.2" / "step1" / "v5_seed1" / "model.pt"}


def main() -> None:
    cfg = run_v5.load_config(paths.REPO_ROOT / "configs" / "v5.yaml")
    rows = []
    for name, file in MODELS.items():
        model = train_v5.load(file, cfg["train"]["dropout"])
        for dataset, takes in run_v5.evaluation_sets("val").items():
            summary, _ = run_v6.argmax_rows(dataset, [run_v5.evidence(model, take, cfg) for take in takes])
            rows.append({"model": name, **summary[0]})
            print(f"{name}  {dataset:>9}  argmax per beat  WCSR {100 * summary[0]['wcsr']:6.2f}%", flush=True)
        mean = float(np.mean([r["wcsr"] for r in rows if r["model"] == name]))
        rows.append({"model": name, "dataset": "mean", "system": run_v6.ARGMAX, "wcsr": mean})
        print(f"{name}       mean  argmax per beat  WCSR {100 * mean:6.2f}%", flush=True)
    out = paths.REPO_ROOT / "results" / "v6" / "v5_argmax.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": "val", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
