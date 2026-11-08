"""Which tuning estimate V11's training cache was built with, for every RWC-P and GuitarSet train and val song.

    python scripts/v12_tuning.py            # -> results/v12/tuning/{table.md, estimates.json}
    python scripts/v12_tuning.py --table    # table.md again from estimates.json

The cache (data_v7_2.cache_dir of configs/v11.yaml) stores the log-CQT, not the tuning. Per song: the
start estimate on the causal harmonic part of the first tuning_seconds, at 24 bins per octave (before
V10, wraps at +/-25 cents) and at 12 bins (features_v7_1.tuning_in_bins, +/-50 cents), both in cents;
then features_v7_1's log-CQT recomputed at each of the two estimates and compared with the cached array
(mean absolute difference). The cache was built with the estimate whose features it matches. ChoCo is
not here: its cache uses the manifest's tuning (data_choco).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np

import run_v5
from chordtracker import data_v5, data_v7_2, features_v7_1, paths

THRESHOLD_CENTS = 3.0
CONFIG = paths.REPO_ROOT / "configs" / "v11.yaml"
OUT = paths.REPO_ROOT / "results" / "v12" / "tuning"


def log_cqt_at(y: np.ndarray, cfg: dict, tuning: float) -> np.ndarray:
    """features_v7_1.causal_log_cqt_of_audio with the start tuning given (bins) instead of estimated."""
    q, comp = cfg["cqt"], cfg["compression"]
    y_h = features_v7_1.causal_harmonic(y, cfg["audio"]["sr"], cfg)
    C = features_v7_1.delayed_cqt(y_h, cfg, 0.0)
    first = int(np.ceil(int(q["tuning_seconds"] * cfg["audio"]["sr"]) / q["hop_length"])) - 1
    C[:, first:] = features_v7_1.delayed_cqt(y_h, cfg, tuning)[:, first:]
    level = features_v7_1.running_level(C, comp["level_percentile"])
    return np.log1p(comp["gamma"] * C / level[None, :]).astype(np.float32)


def matched(diff_old: float, diff_new: float, tol: float = 1e-3) -> str:
    """Which estimate the cache matches (float16 storage: about 1e-4): 'both' when the two features agree with it."""
    if diff_old <= tol and diff_new <= tol:
        return "both"
    if min(diff_old, diff_new) > tol:
        return "neither"
    return "24 bins" if diff_old < diff_new else "12 bins"


def check(job) -> dict:
    take, split, dataset, cfg = job
    sr, q = cfg["audio"]["sr"], cfg["cqt"]
    y, _ = librosa.load(str(take.audio), sr=sr, mono=cfg["audio"]["mono"])
    y_h = features_v7_1.causal_harmonic(y[:int(q["tuning_seconds"] * sr)], sr, cfg)
    bpo = q["bins_per_octave"]
    old = float(librosa.estimate_tuning(y=y_h, sr=sr, bins_per_octave=bpo))
    new = features_v7_1.tuning_in_bins(y_h, sr, bpo)
    cached = np.load(data_v7_2.cache_dir(cfg) / f"{take.key}.npy").astype(np.float32)
    d_old = float(np.abs(log_cqt_at(y, cfg, old) - cached).mean())
    d_new = d_old if abs(new - old) < 1e-9 else float(np.abs(log_cqt_at(y, cfg, new) - cached).mean())
    cents = 1200 / bpo
    return {"key": take.key, "set": dataset, "split": split, "old_cents": old * cents, "new_cents": new * cents,
            "diff_old": d_old, "diff_new": d_new, "cache": matched(d_old, d_new)}


def main() -> None:
    cfg = run_v5.load_config(CONFIG)
    jobs = [(t, split, dataset, cfg) for split in ("train", "val")
            for dataset, takes in (("rwc", data_v5.rwc_takes(split)), ("guitarset", data_v5.guitarset_takes(split)))
            for t in takes]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(check, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "estimates.json").write_text(json.dumps(rows, indent=1))
    write_table(rows)


def write_table(rows: list[dict]) -> None:
    for r in rows:
        r["cache"] = matched(r["diff_old"], r["diff_new"])
    counts = {c: sum(r["cache"] == c for r in rows) for c in ("both", "24 bins", "12 bins", "neither")}
    moved = sorted((r for r in rows if abs(r["new_cents"] - r["old_cents"]) > THRESHOLD_CENTS),
                   key=lambda r: -abs(r["new_cents"] - r["old_cents"]))
    lines = [f"{len(rows)} songs (RWC-P and GuitarSet, train + val). Cache matches: "
             + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".",
             f"{len(moved)} songs whose 12-bin and 24-bin estimates differ by more than {THRESHOLD_CENTS:g} cents:", "",
             "| song | split | 24 bins | 12 bins | change | cache vs 24-bin features | cache vs 12-bin features | cache built with |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in moved:
        lines.append(f"| {r['key']} | {r['split']} | {r['old_cents']:+.1f} | {r['new_cents']:+.1f} | "
                     f"{r['new_cents'] - r['old_cents']:+.1f} | {r['diff_old']:.4f} | {r['diff_new']:.4f} | {r['cache']} |")
    lines += ["", f"Largest difference between the cache and its 24-bin features: {max(r['diff_old'] for r in rows):.5f}."]
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if "--table" in sys.argv:           # the table again from estimates.json
        write_table(json.loads((OUT / "estimates.json").read_text()))
    else:
        main()
