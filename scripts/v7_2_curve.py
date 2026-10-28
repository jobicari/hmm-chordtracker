"""V7.2: from every validation run to the best aligned WCSR per total look-ahead -> results/v7_2/val_curve.json,
figures/v7_2/val/curve.png.

    python scripts/v7_2_curve.py

A point is (grid, input look-ahead a, system, decoder lag, p_self). Its total look-ahead is
a + 1 + lag frames (the 1: the harmonic separation's STFT window; lag 0 for frame-argmax and
forward-filter). Its score is the two-seed mean of the aligned WCSR (mean of RWC-P and GuitarSet).
For each total look-ahead the best point is kept, with both seeds' scores and chord changes per
minute. The two-seed rule: a point beats another only if both its seeds beat both of the other's.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chordtracker import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import run_v7_2  # noqa: E402

GRIDS, LOOKAHEADS, MAX_TOTAL = ("F#0", "C2"), (0, 1, 2, 4), 12


def cell_points(grid: str, a: int) -> list[dict]:
    """Every (system, lag, p_self) of a cell with both seeds' aligned WCSR and changes per minute."""
    runs = [run_v7_2.run_dir(grid, a, s) / "val" / "summary.json" for s in (0, 1)]
    if not all(r.exists() for r in runs):
        return []
    summaries = [json.loads(r.read_text()) for r in runs]
    points = []
    for rows in zip(*(s["rows"] for s in summaries)):
        r = rows[0]
        if r["dataset"] != "mean" or r["lag"] is None:
            continue
        changes = [np.mean([x["mean_changes_per_min"] for x in s["rows"]
                            if x["dataset"] != "mean" and (x["system"], x["p_self"], x["lag"]) ==
                            (r["system"], r["p_self"], r["lag"])]) for s in summaries]
        points.append({"cqt_from": grid, "input_lookahead": a, "system": r["system"], "lag": r["lag"],
                       "p_self": r["p_self"], "total": a + 1 + r["lag"],
                       "seeds": [100 * x["aligned"] for x in rows], "aligned": 100 * float(np.mean([x["aligned"] for x in rows])),
                       "changes_per_min": float(np.mean(changes))})
    return points


def main() -> None:
    cells = {(g, a): cell_points(g, a) for g in GRIDS for a in LOOKAHEADS}
    curve = {}
    for total in range(1, MAX_TOTAL + 1):
        at = [p for pts in cells.values() for p in pts if p["total"] == total]
        if at:
            curve[total] = max(at, key=lambda p: (p["aligned"], -p["changes_per_min"]))
    per_cell = {f"{g} a={a}": {t: max((p for p in pts if p["total"] == t), key=lambda p: p["aligned"])
                               for t in range(1, MAX_TOTAL + 1) if any(p["total"] == t for p in pts)}
                for (g, a), pts in cells.items() if pts}
    out = paths.REPO_ROOT / "results" / "v7_2" / "val_curve.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": curve, "per_cell": per_cell}, indent=2))
    for t, p in curve.items():
        print(f"total {t:2d} ({t * 0.0929:.2f} s): {p['aligned']:.2f}  {p['cqt_from']} a={p['input_lookahead']} "
              f"{p['system']} lag {p['lag']} p_self {p['p_self']}  seeds {[round(s, 2) for s in p['seeds']]}  "
              f"{p['changes_per_min']:.0f} changes/min")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    hop = 2048 / 22050
    colours = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 4: "tab:red"}
    for name, pts in per_cell.items():
        grid, a = name.split(" a=")
        totals = sorted(pts)
        ax.plot(np.array(totals) * hop, [pts[t]["aligned"] for t in totals], marker="o", ms=3, lw=1,
                color=colours[int(a)], linestyle="-" if grid == "F#0" else "--",
                label=f"CQT from {'F#0' if grid == 'F#0' else 'C2'}, input look-ahead {a}")
    totals = sorted(curve)
    reach = np.maximum.accumulate([curve[t]["aligned"] for t in totals])     # a longer delay can always wait
    ax.step(np.array(totals) * hop, reach, "k-", where="post", lw=2.5, alpha=0.35,
            label="best at or below this look-ahead")
    ax.set_xlabel("total look-ahead, s (input look-ahead + 1 frame + decoder lag); playback delayed at least this much")
    ax.set_ylabel("aligned WCSR, %, mean of RWC-P and GuitarSet, two seeds")
    ax.set_title("V7.2 on validation: accuracy against total look-ahead")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig_out = paths.REPO_ROOT / "figures" / "v7_2" / "val" / "curve.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
