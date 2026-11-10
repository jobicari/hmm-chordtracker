"""V15 census: how much diminished, augmented, dim7 and half-diminished there is, per set and split.

    python scripts/v15_census.py      # -> results/v15/census.md

Read from the annotations only (no audio), train and val (test is never read). A label is read as
data_v8 reads it: the bass is ignored and extensions above the octave are dropped (mir_eval.chord.encode
with reduce_extended_chords=False), then the notes up to the 7th decide its group:

    dim      C Eb Gb          C:dim, C:dim/b3, C:dim(9)
    aug      C E G#           C:aug, C:aug(9)
    dim7     C Eb Gb A        C:dim7, C:dim7(b9)
    hdim7    C Eb Gb Bb       C:hdim7, C:dim(b7), C:min7(b5,*5)

and, counted apart because their notes are not one of the four: min(b5) (C Eb Gb G, 'C:min(b5)'),
min7(b5) (C Eb Gb G Bb: mir_eval's 'C:min7(b5)' adds Gb and keeps G), aug(b7) (C E G# Bb), maj(#5)
(C E G G#), 7(#5) and 7(b5); labels mir_eval cannot read whose text names dim or aug (e.g. 'C:aug7') are
listed as unreadable. Per group: seconds, segments, songs, the seconds of inversions (bass not the root)
and the most common label spellings. Context: the labelled seconds and the seconds vocabulary A + N
trains on (data_v8.target with with_n), per set and split.
"""
import sys
from collections import Counter
from pathlib import Path

import mir_eval
import numpy as np

from chordtracker import data_choco, data_v5, data_v8, paths

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config)

OUT = paths.REPO_ROOT / "results" / "v15" / "census.md"
GROUPS = {
    "dim": (0, 3, 6), "aug": (0, 4, 8), "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10),
    "min(b5)": (0, 3, 6, 7), "min7(b5)": (0, 3, 6, 7, 10), "aug(b7)": (0, 4, 8, 10), "maj(#5)": (0, 4, 7, 8),
    "7(#5)": (0, 4, 7, 8, 10), "7(b5)": (0, 4, 6, 7, 10),
}
NEW = ("dim", "aug", "dim7", "hdim7")
SETS = ("RWC-P", "GuitarSet", "ChoCo")


def group(label: str) -> tuple[str | None, bool]:
    """(the label's group or 'unreadable' or None, whether it is an inversion)."""
    try:
        root, bitmap, bass = mir_eval.chord.encode(label, reduce_extended_chords=False)
    except mir_eval.chord.InvalidChordException:
        text = label.split(":", 1)[-1]
        return ("unreadable" if ("dim" in text or "aug" in text) else None), False
    if root < 0:
        return None, False
    notes = tuple(int(i) for i in np.nonzero(bitmap)[0])
    for name, want in GROUPS.items():
        if notes == want:
            return name, bass != 0
    return None, False


def census(takes) -> dict:
    out = {"songs": len(takes), "labelled": 0.0, "trained": 0.0, "groups": {}}
    for take in takes:
        for (s, e), label in zip(take.annotation.intervals, take.annotation.labels):
            d = float(e - s)
            out["labelled"] += d
            if data_v8.target(label, "A", with_n=True) >= 0:
                out["trained"] += d
            g, inverted = group(label)
            if g is None:
                continue
            c = out["groups"].setdefault(g, {"s": 0.0, "segments": 0, "songs": set(), "inv_s": 0.0,
                                              "spellings": Counter()})
            c["s"] += d
            c["segments"] += 1
            c["songs"].add(take.key)
            c["inv_s"] += d if inverted else 0.0
            c["spellings"][label.split(":", 1)[-1]] += 1
    return out


def main() -> None:
    cfg = run_v5.load_config(paths.REPO_ROOT / "configs" / "v8.yaml")
    takes_of = {"RWC-P": data_v5.rwc_takes, "GuitarSet": data_v5.guitarset_takes,
                "ChoCo": lambda split: data_choco.choco_takes(split, cfg)}
    c = {(name, split): census(takes_of[name](split)) for name in SETS for split in ("train", "val")}

    lines = ["# V15 census: dim, aug, dim7, hdim7", "",
             "Seconds from the annotations, train and val only. Groups as scripts/v15_census.py reads the labels "
             "(bass ignored, extensions above the octave dropped). Cells: seconds / segments / songs.", "",
             "## Context", "", "| set | split | songs | labelled s | A + N trained s |", "|---|---|---:|---:|---:|"]
    for (name, split), x in c.items():
        lines.append(f"| {name} | {split} | {x['songs']} | {x['labelled']:.0f} | {x['trained']:.0f} |")
    for split in ("train", "val"):
        tot = sum(c[(n, split)]["trained"] for n in SETS)
        lines.append(f"| all | {split} | {sum(c[(n, split)]['songs'] for n in SETS)} | "
                     f"{sum(c[(n, split)]['labelled'] for n in SETS):.0f} | {tot:.0f} |")

    names = list(GROUPS) + ["unreadable"]
    for split in ("train", "val"):
        lines += ["", f"## {split}", "", "| group | " + " | ".join(SETS) + " | all s | share of A + N trained | "
                  "inversion s |", "|---|" + "---:|" * (len(SETS) + 3)]
        trained = sum(c[(n, split)]["trained"] for n in SETS)
        for g in names:
            cells, total, inv = [], 0.0, 0.0
            for n in SETS:
                x = c[(n, split)]["groups"].get(g)
                cells.append(f"{x['s']:.0f} / {x['segments']} / {len(x['songs'])}" if x else "-")
                total += x["s"] if x else 0.0
                inv += x["inv_s"] if x else 0.0
            if total:
                lines.append(f"| {g} | " + " | ".join(cells) + f" | {total:.0f} | {100 * total / trained:.2f} % | "
                             f"{inv:.0f} |")
    lines += ["", "## Spellings (train and val, all sets)", "", "| group | most common spellings (count of segments) |",
              "|---|---|"]
    for g in names:
        sp = Counter()
        for x in c.values():
            if g in x["groups"]:
                sp.update(x["groups"][g]["spellings"])
        if sp:
            lines.append(f"| {g} | " + ", ".join(f"{k} ({v})" for k, v in sp.most_common(8)) + " |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
