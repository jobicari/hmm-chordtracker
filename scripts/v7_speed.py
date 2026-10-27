"""V7 speed check: the cost of one live frame (CRNN step + decoder step) against the 93 ms between frames.

    python scripts/v7_speed.py results/v6/causal_seed1     # -> results/v7/speed.json

Feeds the log-CQT of the first RWC-P val song to StreamingCRNN one column at a time on the CPU, and
each posterior to FixedLagViterbi (the slowest decoder, at the longest lag of the config). Reports the
mean and 99th-percentile time per frame and the real-time factor (time per frame / hop time; below 1
keeps up with the music). Feature extraction (CQT) is not timed: V7 does not change it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from chordtracker import data_v5, online_hmm, paths, train_v6, transitions
from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS
from chordtracker.streaming_crnn import StreamingCRNN

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402


def main(run: Path) -> None:
    cfg = run_v5.load_config(paths.REPO_ROOT / "configs" / "v7.yaml")
    model = train_v6.load(run / "model.pt", False, cfg["train"]["dropout"])
    prior = np.array(json.loads((run / "prior.json").read_text()))
    A = transitions.transition(run_v5.change_prior(cfg), cfg["decoder"]["p_self_grid"][0])
    song = data_v5.load(data_v5.rwc_takes("val")[0], cfg)
    columns = song.cqt[CENTRE_ROW:CENTRE_ROW + PATCH_BINS].T

    crnn, decoder = StreamingCRNN(model), online_hmm.FixedLagViterbi(A, max(cfg["decoder"]["lags"]))
    seconds = []
    for column in columns:
        start = time.perf_counter()
        post = crnn.step(np.ascontiguousarray(column))
        if post is not None:
            decoder.step(online_hmm.evidence(post[:, None], prior)[:, 0])
        seconds.append(time.perf_counter() - start)
    hop_s = cfg["cqt"]["hop_length"] / cfg["audio"]["sr"]
    ms = 1000 * np.array(seconds[20:])                                  # skip the warm-up frames
    result = {"song": song.key, "frames": len(ms), "mean_ms": float(ms.mean()),
              "p99_ms": float(np.percentile(ms, 99)), "hop_ms": 1000 * hop_s,
              "real_time_factor": float(ms.mean() / (1000 * hop_s))}
    out = paths.REPO_ROOT / "results" / cfg["version"] / "speed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
