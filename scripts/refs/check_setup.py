"""Every tool on one val song, before the full run. BTC: the installed librosa's CQT against the checkpoint's stats."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import score_refs as s

import torch
import yaml

sys.path.insert(0, str(s.BTC_DIR))
from utils.hparams import HParams
from utils.mir_eval_modules import audio_file_to_features

take = s.run_v5.evaluation_sets("val")["guitarset"][0]
checkpoint = torch.load(s.BTC_DIR / "test" / "btc_model.pt", map_location="cpu", weights_only=False)
print("BTC checkpoint mean, std:", float(np.mean(checkpoint["mean"])), float(np.mean(checkpoint["std"])))
config = HParams(**yaml.safe_load((s.BTC_DIR / "run_config.yaml").read_text()))
for t in (take, s.run_v5.evaluation_sets("val")["rwc"][0]):
    feature, time_unit, length = audio_file_to_features(str(t.audio), config)
    print(f"{t.key}: {feature.shape} log-CQT, {feature.shape[1] * time_unit:.2f} s of frames "
          f"for {length:.2f} s of audio, mean {feature.mean():.3f}, std {feature.std():.3f}")

for tool, make in s.RUNNERS.items():
    try:
        intervals, raw = make()(take.audio)
        print(tool, "ok:", s.score_song(take, intervals, raw)["wcsr"], sorted(set(raw))[:12])
    except Exception as error:
        print(tool, "FAILED:", type(error).__name__, error)
