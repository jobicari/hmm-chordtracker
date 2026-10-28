"""V7.1 step 1: the tuning of every training song, to choose between tuning 0 and an estimate from the start.

    python scripts/v7_1_tuning.py          # -> results/v7_1/tuning_train.json

V5 and V6 estimate each file's tuning on the whole (harmonic) audio and shift the CQT filters by it.
A causal system cannot. This measures the whole-file estimate of every training song, exactly as
features_v5.log_cqt computes it, in cents (1 bin = 50 cents at 24 bins per octave). The rule fixed
in the plan before measuring: if 95 % of the songs are within 10 cents of A440, V7.1 uses tuning 0;
otherwise tuning from the first 10 s. Training songs only: validation and test play no part.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

from chordtracker import data_v5, paths
from chordtracker.features import load_audio

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config)

LIMIT_CENTS, SHARE = 10.0, 0.95


def main() -> None:
    cfg = run_v5.load_config(paths.REPO_ROOT / "configs" / "v5.yaml")
    bins = cfg["cqt"]["bins_per_octave"]
    cents_per_bin = 1200 / bins
    songs = {}
    for take in data_v5.rwc_takes("train") + data_v5.guitarset_takes("train"):
        y, sr = load_audio(take.audio, cfg)
        songs[take.key] = cents_per_bin * float(librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=bins))
        print(f"{take.key}: {songs[take.key]:+.1f} cents", flush=True)
    cents = np.array(list(songs.values()))
    share = float(np.mean(np.abs(cents) <= LIMIT_CENTS))
    result = {"songs": songs, "n": len(cents), "share_within_limit": share, "limit_cents": LIMIT_CENTS,
              "median_abs_cents": float(np.median(np.abs(cents))), "max_abs_cents": float(np.abs(cents).max()),
              "choice": "tuning 0" if share >= SHARE else "tuning from the first 10 s"}
    out = paths.REPO_ROOT / "results" / "v7_1" / "tuning_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k != "songs"})


if __name__ == "__main__":
    main()
