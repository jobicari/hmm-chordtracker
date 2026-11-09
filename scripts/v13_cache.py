"""V13: the RWC-P and GuitarSet log-CQT cache again, with the 12-bin tuning estimate (issue #11).

    python scripts/v13_cache.py      # builds data_v7_2.cache_dir of configs/v13.yaml, then checks two songs

The cache V8 to V11 trained on was built before the fix and is kept aside as
data/processed/v7_2_logcqt_Fs1_24bin_tuning; data_v7_2.load computes every missing song with the
current features_v7_1 (12-bin estimate). ChoCo's cache uses its manifest's tuning and is not rebuilt.
The check: for the two songs whose estimate moved most (results/v12/tuning), the new cache equals
features made at the 12-bin estimate (scripts/v12_tuning.log_cqt_at) and differs from the old cache.
"""
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np

import run_v5
import v12_tuning
from chordtracker import data_v5, data_v7_2, features_v7_1, paths

CONFIG = paths.REPO_ROOT / "configs" / "v13.yaml"
OLD = paths.DATA_DIR / "processed" / "v7_2_logcqt_Fs1_24bin_tuning"
CHECK = ("rwc/RWC_P038", "guitarset/02_Rock1-130-A_comp")


def build(take) -> str:
    data_v7_2.load(take, run_v5.load_config(CONFIG))
    return take.key


def main() -> None:
    cfg = run_v5.load_config(CONFIG)
    takes = [t for split in ("train", "val") for t in data_v5.rwc_takes(split) + data_v5.guitarset_takes(split)]
    with ProcessPoolExecutor(8) as pool:
        for i, key in enumerate(pool.map(build, takes), 1):
            if i % 20 == 0 or i == len(takes):
                print(f"{i}/{len(takes)} cached", flush=True)
    sr, q = cfg["audio"]["sr"], cfg["cqt"]
    by_key = {t.key: t for t in takes}
    for key in CHECK:
        y, _ = librosa.load(str(by_key[key].audio), sr=sr, mono=cfg["audio"]["mono"])
        y_h = features_v7_1.causal_harmonic(y[:int(q["tuning_seconds"] * sr)], sr, cfg)
        new_tuning = features_v7_1.tuning_in_bins(y_h, sr, q["bins_per_octave"])
        cached = np.load(data_v7_2.cache_dir(cfg) / f"{key}.npy").astype(np.float32)
        old = np.load(OLD / f"{key}.npy").astype(np.float32)
        expected = v12_tuning.log_cqt_at(y, cfg, new_tuning)
        print(f"{key}: 12-bin estimate {new_tuning * 1200 / q['bins_per_octave']:+.1f} cents; new cache vs 12-bin "
              f"features {np.abs(cached - expected).mean():.5f}, vs old cache {np.abs(cached - old).mean():.5f}")


if __name__ == "__main__":
    main()
