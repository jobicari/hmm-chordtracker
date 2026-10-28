"""V7.1 validation check: where the causal features lose accuracy, against V7 (V6's causal CRNN).

    python scripts/v7_1_breakdown.py          # -> results/v7_1/val_breakdown.json

Frame accuracy of the network alone (argmax per frame, frames with a target), both seeds of each
system, pooled over each validation set, split two ways:
    start        the first 5 s of a song (the running level and the GRU state are still settling) / the rest
    after change the first 0.5 s after a reference chord change (the causal median holds new notes back
                 about 0.37 s) / the rest
Validation only.
"""
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import acoustic_hmm, data_v5, data_v7_1, paths, train_v6, train_v7_1

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402

START_S, AFTER_S = 5.0, 0.5
SYSTEMS = {
    "v7": dict(config="v7.yaml", runs=["results/v6/causal_seed0", "results/v6/causal_seed1"],
               load=data_v5.load, posteriors=train_v6.song_posteriors),
    "v7_1 median 31": dict(config="v7_1.yaml", runs=["results/v7_1/seed0", "results/v7_1/seed1"], median=31,
                           load=data_v7_1.load, posteriors=train_v7_1.song_posteriors),
    "v7_1 median 9": dict(config="v7_1.yaml", runs=["results/v7_1/median9_seed0", "results/v7_1/median9_seed1"],
                          median=9, load=data_v7_1.load, posteriors=train_v7_1.song_posteriors),
}


def masks(take, times):
    """Boolean frame masks: within the first START_S s, within AFTER_S s after a reference chord change."""
    starts = take.annotation.intervals[1:, 0]
    changed = np.array([take.annotation.labels[i + 1] != take.annotation.labels[i]
                        for i in range(len(starts))], dtype=bool)
    since = times[:, None] - starts[changed][None, :]
    return times < START_S, ((since >= 0) & (since < AFTER_S)).any(axis=1)


def main() -> None:
    out = {}
    for name, s in SYSTEMS.items():
        cfg = run_v5.load_config(paths.REPO_ROOT / "configs" / s["config"])
        if "median" in s:
            cfg["audio"]["harmonic_median_frames"] = s["median"]
        for run in s["runs"]:
            model = train_v6.load(paths.REPO_ROOT / run / "model.pt", False, 0.5)
            for dataset, takes in run_v5.evaluation_sets("val").items():
                counts = {k: [0, 0] for k in ("start", "rest_start", "after_change", "rest_change")}
                for take in takes:
                    song = s["load"](take, cfg)
                    right = s["posteriors"](model, song).argmax(axis=0) == song.targets
                    keep = song.targets >= 0
                    times = acoustic_hmm.frame_times(len(right), cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
                    start, after = masks(take, times)
                    for key, m in (("start", start), ("rest_start", ~start), ("after_change", after),
                                   ("rest_change", ~after)):
                        counts[key][0] += int((right & keep & m).sum())
                        counts[key][1] += int((keep & m).sum())
                out[f"{name} {run} {dataset}"] = {k: r / n for k, (r, n) in counts.items()}
                print(name, run, dataset, {k: round(100 * v, 2) for k, v in out[f"{name} {run} {dataset}"].items()},
                      flush=True)
    file = paths.REPO_ROOT / "results" / "v7_1" / "val_breakdown.json"
    file.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
