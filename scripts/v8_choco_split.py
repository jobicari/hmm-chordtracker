"""V8: the ChoCo export's train / val / test split, grouped so no album or artist is in two splits.

    python scripts/v8_choco_split.py <export dir> [--seed 0] > splits/choco.json

Groups: isophonics and robbie-williams by release (The Beatles CD1 and CD2 are one album), uspop2002
by artist. jaah (9 recordings, one per artist) is too small to split and goes to train whole.
Each partition's groups are shuffled with the seed and dealt, largest first, to whichever split is
furthest below its share of that partition's duration (70 / 15 / 15), so each split holds every
partition in about those proportions. Standard library only: it reads the manifests and JAMS as JSON.
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}
GROUP_BY = {"isophonics": "release", "robbie-williams": "release", "uspop2002": "artist"}
TRAIN_ONLY = ("jaah",)


def recordings(export: Path, partition: str) -> list[dict]:
    """Manifest rows of a partition, each with its release from the JAMS file metadata."""
    rows = list(csv.DictReader(open(export / partition / "manifest.csv")))
    for row in rows:
        meta = json.loads((export / partition / row["jams"]).read_text())["file_metadata"]
        row["release"] = meta.get("release") or ""
    return rows


def group_key(row: dict, partition: str) -> str:
    """Album or artist of a recording, lower case; 'The Beatles CD1' and 'CD2' are one album."""
    value = row[GROUP_BY[partition]]
    return re.sub(r"\s+cd\s*\d+$", "", value.strip().lower())


def deal(groups: dict[str, list[dict]], seed: int) -> dict[str, list[str]]:
    """Groups shuffled, then largest first to the split furthest below its duration share."""
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -sum(float(r["duration"]) for r in groups[k]))   # stable: ties keep the shuffle
    total = sum(float(r["duration"]) for g in groups.values() for r in g)
    held = {s: 0.0 for s in SHARES}
    out = {s: [] for s in SHARES}
    for k in keys:
        split = max(SHARES, key=lambda s: SHARES[s] * total - held[s])
        out[split] += [r["id"] for r in groups[k]]
        held[split] += sum(float(r["duration"]) for r in groups[k])
    return out


def make_split(export: Path, seed: int = 0) -> dict:
    split = {"seed": seed, "train": [], "val": [], "test": []}
    for partition in GROUP_BY:
        groups: dict[str, list[dict]] = {}
        for row in recordings(export, partition):
            groups.setdefault(group_key(row, partition), []).append(row)
        for s, ids in deal(groups, seed).items():
            split[s] += ids
    for partition in TRAIN_ONLY:
        split["train"] += [r["id"] for r in csv.DictReader(open(export / partition / "manifest.csv"))]
    for s in SHARES:
        split[s].sort(key=lambda i: (i.rsplit("_", 1)[0], int(i.rsplit("_", 1)[1])))
    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    json.dump(make_split(args.export, args.seed), sys.stdout, indent=1)
    print()
