"""V7.2 validation check: frame accuracy of the network alone right after chord changes, by input look-ahead.

    python scripts/v7_2_breakdown.py          # -> results/v7_2/val_breakdown.json

As scripts/v7_1_breakdown.py (first 0.5 s after a reference chord change / the rest of the song),
for every V7.2 cell and both seeds. Validation only.
"""
import json
import sys
from pathlib import Path

from chordtracker import acoustic_hmm, data_v7_2, paths, train_v7_2

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402
import run_v7_2  # noqa: E402
from v7_1_breakdown import masks  # noqa: E402


def main() -> None:
    base = run_v5.load_config(paths.REPO_ROOT / "configs" / "v7_2.yaml")
    out = {}
    for grid in ("F#0", "C2"):
        for a in (0, 1, 2, 4):
            for seed in (0, 1):
                cfg = run_v7_2.cell_config(base, grid, a, seed)
                model = train_v7_2.load(run_v7_2.run_dir(grid, a, seed) / "model.pt", False, 0.5)
                for dataset, takes in run_v5.evaluation_sets("val").items():
                    counts = {k: [0, 0] for k in ("after_change", "rest")}
                    for take in takes:
                        song = data_v7_2.load(take, cfg)
                        right = train_v7_2.song_posteriors(model, song, a).argmax(axis=0) == song.targets
                        keep = song.targets >= 0
                        times = acoustic_hmm.frame_times(len(right), cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
                        after = masks(take, times)[1]
                        for key, m in (("after_change", after), ("rest", ~after)):
                            counts[key][0] += int((right & keep & m).sum())
                            counts[key][1] += int((keep & m).sum())
                    key = f"{grid} a={a} seed{seed} {dataset}"
                    out[key] = {k: r / n for k, (r, n) in counts.items()}
                    print(key, {k: round(100 * v, 2) for k, v in out[key].items()}, flush=True)
    (paths.REPO_ROOT / "results" / "v7_2" / "val_breakdown.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
