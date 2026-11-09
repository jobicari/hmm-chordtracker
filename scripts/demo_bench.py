"""How the live chain keeps up on this machine: the demo server's work, timed on the prepared samples.

    python scripts/demo_bench.py                     # natively (PYTHONPATH=src): each sample alone
    python scripts/demo_bench.py --sessions 8        # 8 listeners at once, at playback speed
    docker run --rm chord-tracker-demo python scripts/demo_bench.py --sessions 8      # in the image

Alone: each prepared sample (demo/samples) runs through a fresh session (play_along.Analysis, as the
server runs it) one frame of 2,048 samples (93 ms of audio) at a time, as fast as it goes. Per sample:
the median, p99 and worst time of a step, the CPU it takes as a fraction of real time, and the worst lag
(the step times played as a queue at playback speed: how long after a frame's audio has arrived its
labels are out; a slow step delays the frames behind it until the queue drains).

--sessions N: N sessions in one thread pool, as the server runs them, each step when its audio would have
arrived (the first sample, every session started 0.5 s after the previous one); the lag is measured.
The server is one Python process and much of a step holds Python's interpreter lock, so its sessions
share about one CPU whatever the machine has (measured on a 10-core M1 Pro: 2 sessions at once ran 1.4
times as fast as one after the other, 8 no faster): more listeners need more containers, not bigger ones.

Then the suggested CHORD_DEMO_MAX_SESSIONS (half of one CPU at the CPU per listener measured alone) and
CHORD_DEMO_LIVE_MARGIN (the worst lag plus 0.1 s for the network, at least the default): the page adds
the margin to its playback delay for this lag and the network. Exits 1 if the worst lag is beyond this
server's margin: chords would then come later than the page's delay allows.
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import demo_server  # noqa: E402

from chordtracker import paths  # noqa: E402

SAMPLES = paths.REPO_ROOT / "demo" / "samples"
NETWORK_S = 0.1                 # what the suggested margin leaves for the network round trip
STAGGER_S = 0.5                 # --sessions: listeners do not all press Play at the same instant


def step_times(session, y: np.ndarray, block: int) -> np.ndarray:
    """Seconds each session.step took, block by block, as fast as it goes."""
    out = []
    for i in range(0, len(y), block):
        t = time.perf_counter()
        session.step(y[i:i + block])
        out.append(time.perf_counter() - t)
    return np.array(out)


def lags(times: np.ndarray, block_seconds: float) -> np.ndarray:
    """How long after block i's audio has arrived ((i + 1) x block_seconds) its step ends, one step at a time.

    Example: blocks of 0.1 s taking [0.05, 0.3, 0.05, 0.05] -> lags [0.05, 0.3, 0.25, 0.2].
    """
    out, free = [], 0.0
    for i, t in enumerate(times):
        arrival = (i + 1) * block_seconds
        free = max(free, arrival) + t
        out.append(free - arrival)
    return np.array(out)


def paced_lags(session, y: np.ndarray, block: int, sr: int, start: float) -> np.ndarray:
    """Each block stepped when its audio would have arrived (playback from `start`, perf_counter time); lags, s."""
    out = []
    for i in range(0, len(y), block):
        arrival = start + min(i + block, len(y)) / sr
        time.sleep(max(0.0, arrival - time.perf_counter()))
        session.step(y[i:i + block])
        out.append(time.perf_counter() - arrival)
    return np.array(out)


def suggestions(cpu_fraction: float, worst_lag: float, default_margin: float) -> tuple[int, float]:
    """(max sessions, live margin s): half of one CPU at cpu_fraction per listener; the worst lag + the network.

    Example: 0.07 of a CPU per listener, worst lag 0.08 s -> (7, 0.2).
    """
    sessions = max(1, math.floor(0.5 / cpu_fraction))
    margin = max(default_margin, math.ceil((worst_lag + NETWORK_S) * 20 - 1e-9) / 20)
    return sessions, margin


def load(sample_id: str, sr: int) -> np.ndarray:
    meta = json.loads((SAMPLES / f"{sample_id}.json").read_text())
    return librosa.load(str(SAMPLES / meta["audio"]), sr=sr, mono=True)[0]    # as the page resamples it


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("samples", nargs="*", help="sample ids in demo/samples (default: all)")
    parser.add_argument("--sessions", type=int, default=0, help="also run this many listeners at once")
    args = parser.parse_args()
    ids = args.samples or [s["id"] for s in json.loads((SAMPLES / "index.json").read_text())["samples"]]

    import torch
    torch.set_num_threads(1)                                     # as demo_server.serve
    chain = demo_server.Chain()
    block, sr = chain.hop, chain.sr
    frame_s = block / sr
    warm = chain.new_session()                                   # numba compilation, the tuning step once
    warm.step(load(ids[0], sr))
    warm.finish()

    print(f"one step = {block} samples = {1000 * frame_s:.0f} ms of audio")
    worst_lag, fractions = 0.0, []
    for sample_id in ids:
        times = step_times(chain.new_session(), load(sample_id, sr), block)
        lag = lags(times, frame_s)
        fraction = times.sum() / (len(times) * frame_s)
        worst_lag, fractions = max(worst_lag, lag.max()), fractions + [fraction]
        print(f"{sample_id}: {len(times)} steps; median {1000 * np.median(times):.1f} ms, "
              f"p99 {1000 * np.percentile(times, 99):.1f} ms, worst {1000 * times.max():.0f} ms "
              f"(step {times.argmax()}, {times.argmax() * frame_s:.1f} s); CPU {100 * fraction:.1f} % of real time; "
              f"worst lag {1000 * lag.max():.0f} ms")

    if args.sessions:
        y = load(ids[0], sr)
        t0 = time.perf_counter() + 0.1
        with ThreadPoolExecutor(max_workers=args.sessions) as pool:
            runs = list(pool.map(lambda k: paced_lags(chain.new_session(), y, block, sr, t0 + k * STAGGER_S),
                                 range(args.sessions)))
        every = np.concatenate(runs)
        worst_lag = max(worst_lag, every.max())
        print(f"{args.sessions} sessions at once ({ids[0]}, at playback speed): lag median "
              f"{1000 * np.median(every):.0f} ms, p99 {1000 * np.percentile(every, 99):.0f} ms, "
              f"worst {1000 * every.max():.0f} ms")

    sessions, margin = suggestions(max(fractions), worst_lag, demo_server.LIVE_MARGIN)
    print(f"suggested: CHORD_DEMO_MAX_SESSIONS={sessions} CHORD_DEMO_LIVE_MARGIN={margin:g} "
          f"(this server: {demo_server.MAX_SESSIONS}, {demo_server.LIVE_MARGIN:g})")
    if worst_lag > demo_server.LIVE_MARGIN:
        print(f"too slow: the worst lag ({worst_lag:.2f} s) is beyond the margin ({demo_server.LIVE_MARGIN:g} s); "
              "chords would come later than the page's delay allows", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
