"""V7.3 speed check: the whole live chain, one 2,048-sample block at a time, against the 93 ms between blocks.

    python scripts/v7_3_speed.py           # -> results/v7_3/speed.json

Feeds the longest RWC-P validation song to StreamingFeatures (C2 grid), the C2 network with input
look-ahead 2 (LookaheadStreamingCRNN) and a fixed-lag Viterbi at lag 4, on the CPU. Times each stage per
block: mean, 99th percentile and maximum (the 10 s tuning estimate is a one-off), and the real-time
factor (mean time per block / 93 ms).
"""
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

from chordtracker import data_v5, online_hmm, paths, train_v7_2, transitions
from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS
from chordtracker.streaming_features import StreamingFeatures
from chordtracker.streaming_v7_2 import LookaheadStreamingCRNN

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402
import run_v7_2  # noqa: E402


def main() -> None:
    base = run_v5.load_config(paths.REPO_ROOT / "configs" / "v7_2.yaml")
    cfg = run_v7_2.cell_config(base, "C2", 2, 0)
    run = run_v7_2.run_dir("C2", 2, 0)
    model = train_v7_2.load(run / "model.pt", False, 0.5)
    prior = np.array(json.loads((run / "prior.json").read_text()))
    take = max(data_v5.rwc_takes("val"), key=lambda t: t.annotation.intervals[-1, 1])
    y, _ = librosa.load(str(take.audio), sr=cfg["audio"]["sr"], mono=True)
    hop = cfg["cqt"]["hop_length"]

    features, crnn = StreamingFeatures(cfg), LookaheadStreamingCRNN(model, 2)
    decoder = online_hmm.FixedLagViterbi(transitions.transition(run_v5.change_prior(cfg), 0.98), 4)
    times = {"features": [], "crnn": [], "decoder": [], "total": []}
    for i in range(0, len(y), hop):
        t0 = time.perf_counter()
        cols = features.step(y[i:i + hop])
        t1 = time.perf_counter()
        posts = [p for p in (crnn.step(c[CENTRE_ROW:CENTRE_ROW + PATCH_BINS]) for c in cols) if p is not None]
        t2 = time.perf_counter()
        for p in posts:
            decoder.step(online_hmm.evidence(p[:, None], prior)[:, 0])
        t3 = time.perf_counter()
        for k, v in zip(times, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
            times[k].append(1000 * v)
    hop_ms = 1000 * hop / cfg["audio"]["sr"]
    result = {"song": take.key, "seconds": len(y) / cfg["audio"]["sr"], "blocks": len(times["total"]), "hop_ms": hop_ms}
    for k, v in times.items():
        v = np.array(v[20:])
        result[k] = {"mean_ms": float(v.mean()), "p99_ms": float(np.percentile(v, 99)), "max_ms": float(v.max()),
                     "last_minute_mean_ms": float(v[-640:].mean())}
    result["real_time_factor"] = result["total"]["mean_ms"] / hop_ms
    out = paths.REPO_ROOT / "results" / "v7_3" / "speed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
