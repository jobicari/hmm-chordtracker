"""V14 census: how much reference N (no chord) there is, per set and split, and where in the songs it lies.

    python scripts/v14_census.py      # -> results/v14/census.md

Read from the annotations only (no audio). Per set (RWC-P, GuitarSet, ChoCo) and split (train, val):
seconds of N, and its share of the labelled time (the time inside the annotation) and of the time a
vocabulary A + N model trains on (N plus chords in vocabulary A). Where N lies: seconds within the first
10 s of a song, within its last 10 s, and in between (mid-song); a song shorter than 20 s counts its
overlap with the edges first.
"""
import sys
from pathlib import Path

import numpy as np

from chordtracker import data_choco, data_v5, data_v8, paths

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config)

EDGE_S = 10.0
OUT = paths.REPO_ROOT / "results" / "v14" / "census.md"


def overlap(a: float, b: float, lo: float, hi: float) -> float:
    return max(0.0, min(b, hi) - max(a, lo))


def census(takes) -> dict:
    """Seconds: labelled, N, A chords, N in the first / last 10 s and mid-song; songs with any N."""
    out = {"songs": len(takes), "songs_with_n": 0, "labelled": 0.0, "n": 0.0, "a": 0.0, "start": 0.0, "end": 0.0,
           "mid": 0.0}
    for take in takes:
        iv, labels = take.annotation.intervals, take.annotation.labels
        first, last = float(iv[:, 0].min()), float(iv[:, 1].max())
        has_n = False
        for (s, e), label in zip(iv, labels):
            out["labelled"] += e - s
            if label == "N":
                has_n = has_n or e > s
                out["n"] += e - s
                head = overlap(s, e, first, first + EDGE_S)
                tail = overlap(s, e, max(first + EDGE_S, last - EDGE_S), last)
                out["start"] += head
                out["end"] += tail
                out["mid"] += (e - s) - head - tail
            elif data_v8.target(label, "A") >= 0:
                out["a"] += e - s
        out["songs_with_n"] += has_n
    return out


def main() -> None:
    cfg = run_v5.load_config(paths.REPO_ROOT / "configs" / "v8.yaml")
    sets = {"RWC-P": data_v5.rwc_takes, "GuitarSet": data_v5.guitarset_takes,
            "ChoCo": lambda split: data_choco.choco_takes(split, cfg)}
    lines = ["# V14 census: reference N (no chord)", "",
             "Seconds from the annotations. Share of labelled: N over the time inside the annotation. Share of "
             "trained: N over N plus the chords vocabulary A trains on (what an A + N model sees). First / last "
             f"{EDGE_S:g} s and mid-song: where the N seconds lie in each song (percent of the N seconds).", "",
             "| set | split | songs | songs with N | labelled s | N s | share of labelled | share of trained | "
             f"first {EDGE_S:g} s | last {EDGE_S:g} s | mid-song |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, takes_of in sets.items():
        for split in ("train", "val"):
            c = census(takes_of(split))
            pct = (lambda x: f"{100 * x / c['n']:.1f} %") if c["n"] else (lambda x: "-")
            lines.append(f"| {name} | {split} | {c['songs']} | {c['songs_with_n']} | {c['labelled']:.0f} | "
                         f"{c['n']:.0f} | {100 * c['n'] / c['labelled']:.2f} % | "
                         f"{100 * c['n'] / (c['n'] + c['a']):.2f} % | {pct(c['start'])} | {pct(c['end'])} | "
                         f"{pct(c['mid'])} |")
            print(lines[-1], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
