"""V7 figure: accuracy against latency on val, mean of the two causal seeds -> figures/v7/val/latency.png.

    python scripts/v7_plot.py results/v7/causal_seed0 results/v7/causal_seed1

For fixed-lag-viterbi at every lag (x = lag in seconds), with the p_self chosen per lag on the
two-seed mean of each score:
    decision  how right the decision for a frame is, however late it comes
    screen    what a live display shows (the decision for frame t - L at frame t)
    wall      screen with the front end's look-ahead added (x shifted by it)
Horizontal lines: frame-argmax (lag 0), offline-frame-viterbi and beat-viterbi-v6 (whole song).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chordtracker import paths  # noqa: E402


def mean_rows(runs: list[Path]) -> dict:
    """{(system, p_self, lag): {kind: two-seed mean}} over the 'mean' rows (RWC-P and GuitarSet)."""
    summaries = [json.loads((run / "val" / "summary.json").read_text()) for run in runs]
    out = {}
    for rows in zip(*(s["rows"] for s in summaries)):
        if rows[0]["dataset"] == "mean":
            out[(rows[0]["system"], rows[0]["p_self"], rows[0]["lag"])] = {
                kind: None if rows[0][kind] is None else 100 * float(np.mean([r[kind] for r in rows]))
                for kind in ("decision", "screen", "wall")}
    return out, summaries[0]["config"]


def main(runs: list[Path]) -> None:
    rows, cfg = mean_rows(runs)
    hop_s = cfg["cqt"]["hop_length"] / cfg["audio"]["sr"]
    lags = cfg["decoder"]["lags"]
    front_s = cfg["front_end_lookahead_frames"] * hop_s
    best = lambda system, lag, kind: max(v[kind] for (s, _, l), v in rows.items() if s == system and l == lag)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    x = np.array(lags) * hop_s
    for ax in axes:
        ax.plot(x, [best("fixed-lag-viterbi", lag, "decision") for lag in lags], "o-",
                label="fixed-lag Viterbi, decision")
        ax.plot(x, [best("fixed-lag-viterbi", lag, "screen") for lag in lags], "s-",
                label="fixed-lag Viterbi, screen")
        ax.plot(x + front_s, [best("fixed-lag-viterbi", lag, "wall") for lag in lags], "^-",
                label="fixed-lag Viterbi, wall (+ input look-ahead)")
        for system, style in (("frame-argmax", ":"), ("offline-frame-viterbi", "--"), ("beat-viterbi-v6", "-.")):
            lag = 0 if system == "frame-argmax" else None
            ax.axhline(best(system, lag, "decision"), color="grey", linestyle=style, linewidth=1,
                       label=f"{system}, decision")
        ax.grid(alpha=0.3)
        ax.set_xlabel("latency, s")
    axes[0].set_ylabel("WCSR, %, mean of RWC-P and GuitarSet, two seeds")
    axes[0].set_title("all lags (wall: + the input's look-ahead)")
    axes[0].legend(fontsize=8, loc="center right")
    axes[1].set_xlim(-0.03, 0.8)
    axes[1].set_ylim(70, 78)
    axes[1].set_title("zoom: lags 0 to 8 frames (0.75 s)")
    fig.suptitle("V7 on val: accuracy against latency, causal CRNN")
    out = paths.REPO_ROOT / "figures" / "v7" / "val" / "latency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
