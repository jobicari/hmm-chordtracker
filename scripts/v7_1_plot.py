"""V7.1 figure: end-to-end accuracy against latency on validation, V7 (V6 features) against V7.1 (causal
features) -> figures/v7_1/val/latency.png.

    python scripts/v7_1_plot.py

For fixed-lag-viterbi at every lag, two-seed means, p_self chosen per lag on each score. x is the
latency from the sound to the label: the decoder's lag plus the input's look-ahead (V7: 17 frames,
V7.1: 1 frame). Solid: end-to-end WCSR (each label scored at that moment). Dashed: aligned WCSR
(each label scored at the frame it describes), drawn at the same latency for comparison.
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chordtracker import paths  # noqa: E402

RUNS = {"V7: V6 features (1.6 s look-ahead)": ("results/v7/causal_seed0", "results/v7/causal_seed1"),
        "V7.1: causal features (1 frame)": ("results/v7_1/median9_seed0", "results/v7_1/median9_seed1")}


def curve(runs):
    """{lag: (best two-seed aligned, best two-seed end-to-end)} for fixed-lag-viterbi, and the config."""
    summaries = [json.loads((paths.REPO_ROOT / r / "val" / "summary.json").read_text()) for r in runs]
    best = {}
    for rows in zip(*(s["rows"] for s in summaries)):
        r = rows[0]
        if r["dataset"] == "mean" and r["system"] == "fixed-lag-viterbi":
            a, e = (100 * float(np.mean([x[k] for x in rows])) for k in ("aligned", "end_to_end"))
            old = best.get(r["lag"], (0, 0))
            best[r["lag"]] = (max(old[0], a), max(old[1], e))
    return best, summaries[0]["config"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, runs), colour in zip(RUNS.items(), ("tab:grey", "tab:blue")):
        best, cfg = curve(runs)
        hop_s = cfg["cqt"]["hop_length"] / cfg["audio"]["sr"]
        lags = sorted(best)
        x = (np.array(lags) + cfg["front_end_lookahead_frames"]) * hop_s
        ax.plot(x, [best[l][1] for l in lags], "o-", color=colour, label=f"{name}, end-to-end")
        ax.plot(x, [best[l][0] for l in lags], "o--", color=colour, alpha=0.6, label=f"{name}, aligned")
    ax.set_xlabel("latency from the sound to the label, s (decoder lag + input look-ahead)")
    ax.set_ylabel("WCSR, %, mean of RWC-P and GuitarSet, two seeds")
    ax.set_title("V7.1 on validation: fixed-lag Viterbi, V7 against V7.1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    out = paths.REPO_ROOT / "figures" / "v7_1" / "val" / "latency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
