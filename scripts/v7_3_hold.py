"""V7.3 extra: the hold rule against flicker at short delays, on validation.

    python scripts/v7_3_hold.py           # -> results/v7_3/val_hold.json

V7.2's 0.28 s network (C2 grid, input look-ahead 2), both seeds, cached features. For m = 1, 2, 3, 5 the
hold rule (total look-ahead 3 + m - 1 frames) against the fixed-lag Viterbi at lag m - 1 (the same total,
read from results/v7_2/c2_la2_seed*/val, best p_self): aligned WCSR (mean of RWC-P and GuitarSet) and
chord changes per minute. Validation only.
"""
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import acoustic_hmm, data_v7_2, evaluate, hold, paths, train_v7_2

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402
import run_v7  # noqa: E402
import run_v7_2  # noqa: E402

HOLDS = (1, 2, 3, 5)


def main() -> None:
    base = run_v5.load_config(paths.REPO_ROOT / "configs" / "v7_2.yaml")
    out = {}
    for seed in (0, 1):
        cfg = run_v7_2.cell_config(base, "C2", 2, seed)
        run = run_v7_2.run_dir("C2", 2, seed)
        model = train_v7_2.load(run / "model.pt", False, 0.5)
        for m in HOLDS:
            per_set, changes = [], []
            for dataset, takes in run_v5.evaluation_sets("val").items():
                scores = []
                for take in takes:
                    song = data_v7_2.load(take, cfg)
                    post = train_v7_2.song_posteriors(model, song, 2)
                    times = acoustic_hmm.frame_times(post.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
                    s, c = run_v7.score(hold.hold(post.argmax(axis=0), m), times, take)
                    scores.append(s)
                    changes.append(c)
                per_set.append(evaluate.pooled(scores).wcsr)
            out[f"hold m={m} seed{seed}"] = {"total": 3 + m - 1, "aligned": 100 * float(np.mean(per_set)),
                                             "changes_per_min": float(np.mean(changes))}
            print(f"hold m={m} seed{seed}", out[f"hold m={m} seed{seed}"], flush=True)
        summary = json.loads((run / "val" / "summary.json").read_text())
        for m in HOLDS[1:]:
            rows = [r for r in summary["rows"] if r["dataset"] == "mean" and r["system"] == run_v7.FIXED_LAG
                    and r["lag"] == m - 1]
            best = max(rows, key=lambda r: r["aligned"])
            chg = np.mean([r["mean_changes_per_min"] for r in summary["rows"] if r["dataset"] != "mean"
                           and r["system"] == run_v7.FIXED_LAG and r["lag"] == m - 1 and r["p_self"] == best["p_self"]])
            out[f"fixed-lag lag={m - 1} seed{seed}"] = {"total": 3 + m - 1, "aligned": 100 * best["aligned"],
                                                       "p_self": best["p_self"], "changes_per_min": float(chg)}
            print(f"fixed-lag lag={m - 1} seed{seed}", out[f"fixed-lag lag={m - 1} seed{seed}"], flush=True)
    file = paths.REPO_ROOT / "results" / "v7_3" / "val_hold.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
