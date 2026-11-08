"""V10 step 3: the live-path tuning estimate before and after the fix of issue #11, on RWC-P and GuitarSet.

    python scripts/v10_tuning.py      # -> results/v10/step3_tuning/{table.md, estimates.json}

For every train and val song: the estimate the caches were built with (librosa.estimate_tuning at 24 bins
per octave on the causal harmonic part of the first tuning_seconds, -25 .. +25 cents, wrapping) and the
fixed one (features_v7_1.tuning_in_bins: 12 bins, -50 .. +50 cents), both in cents. ChoCo is not here:
its cache uses the manifest's tuning. The table lists the songs whose estimate moves by more than 5 cents.
"""
import json

import librosa
import numpy as np

import run_v5
from chordtracker import data_v5, features_v7_1, paths

THRESHOLD_CENTS = 5.0


def estimates(take, cfg: dict) -> tuple[float, float]:
    sr, q = cfg["audio"]["sr"], cfg["cqt"]
    y, _ = librosa.load(str(take.audio), sr=sr, mono=cfg["audio"]["mono"])
    n = int(q["tuning_seconds"] * sr)
    y_h = features_v7_1.causal_harmonic(y[:n], sr, cfg)
    old = float(librosa.estimate_tuning(y=y_h, sr=sr, bins_per_octave=24)) * 50
    new = features_v7_1.tuning_in_bins(y_h, sr, 24) * 50
    return old, new


def main() -> None:
    cfg = run_v5.load_config(paths.REPO_ROOT / "configs" / "v10.yaml")
    rows = []
    for split in ("train", "val"):
        for dataset, takes in (("rwc", data_v5.rwc_takes(split)), ("guitarset", data_v5.guitarset_takes(split))):
            for take in takes:
                old, new = estimates(take, cfg)
                rows.append({"key": take.key, "set": dataset, "split": split, "old_cents": old, "new_cents": new})
            print(f"{dataset} {split}: {len(takes)} songs", flush=True)
    moved = [r for r in rows if abs(r["new_cents"] - r["old_cents"]) > THRESHOLD_CENTS]
    lines = [f"{len(rows)} songs (RWC-P and GuitarSet, train + val); {len(moved)} move by more than "
             f"{THRESHOLD_CENTS:g} cents.", "", "| song | split | old (24 bins) | new (12 bins) | change |",
             "|---|---|---:|---:|---:|"]
    for r in sorted(moved, key=lambda r: -abs(r["new_cents"] - r["old_cents"])):
        lines.append(f"| {r['key']} | {r['split']} | {r['old_cents']:+.1f} | {r['new_cents']:+.1f} | "
                     f"{r['new_cents'] - r['old_cents']:+.1f} |")
    diff = np.array([r["new_cents"] - r["old_cents"] for r in rows])
    lines += ["", f"Change over all songs: median {np.median(np.abs(diff)):.2f} cents, max {np.abs(diff).max():.2f}."]
    out = paths.REPO_ROOT / "results" / "v10" / "step3_tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / "estimates.json").write_text(json.dumps(rows, indent=1))
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
