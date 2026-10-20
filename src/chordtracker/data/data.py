from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
import random
import numpy as np
import jams
import mir_eval
import csv
import os
import json



@dataclass(frozen=True)
class ChordAnnotation:
    intervals: np.ndarray       # (n, 2) float, seconds
    labels: list[str]           # n Harte labels

    def __post_init__(self):  # runs after __init__: reject bad data at creation
        if self.intervals.shape != (len(self.labels), 2):
            raise ValueError(f"{self.intervals.shape} intervals for {len(self.labels)} labels")

    @classmethod
    def from_lab(cls, path):
        intervals, labels =  mir_eval.io.load_labeled_intervals(str(path))
        return cls(intervals, list(labels))


    @classmethod
    def from_jams(cls, path: Path | str, index=0):
        """index-th chord annotation in the file (GuitarSet: 0 = lead sheet, 1 = performed)"""
        if isinstance(path, Path):
            path = str(path)
        with open(path) as f:
            annos = [a for a in jams.load(f)["annotations"] if a.namespace=="chord"]
        data = annos[index]["data"]
        intervals = np.array([[o.time, o.time+o.duration] for o in data])   # start time, end time
        labels = [o.value for o in data]                                    # chord labels
        return cls(intervals, labels)


    def to_lab(self, path):
        with open(path, "w") as f:
            for (start, end), lab in zip(self.intervals, self.labels):
                f.write(f"{start:.6f}\t{end:.6f}\t{lab}\n")


    def at(self, times):
        """Label at each time; times outside every interval get 'N'."""
        # Checks starts time idx
        idx = np.searchsorted(self.intervals[:,0], times, side="right") - 1
        # Checks if within segments
        # np.clip converts a any negatives, which could only be -1, to 0 (because it would return the last index)
        inside = (idx >= 0) & (times < self.intervals[np.clip(idx, 0, None),1])
        return [self.labels[i] if ok else "N" for i, ok in zip(idx, inside)]

    def __iter__(self):
        # yields (start, end, label), so `for s, e, lab in ann:`
        return zip(map(tuple, self.intervals), self.labels)




@dataclass(frozen=True)
class TakeRecord:         # one row of the manifest
    song_id: str
    group: str      # takes in a group stay in one split
    audio: Path
    lab: Path       # .lab file. Format comes from the Isophonics/Beatles chord annotations by Chris Harte, and MIREX




@dataclass(frozen=True)
class RawTake:            # what an adapter yields, before anything is written
    song_id: str
    group: str
    audio: Path
    chords: ChordAnnotation



# ====================================================================================
# ================================    ADAPTERS      ==================================
# ====================================================================================

SPLITS = ("train", "val", "test")


class DatasetAdapter(ABC):
    """Reads one dataset. Subclasses must implement raw_takes(); split_groups() has a default."""

    @abstractmethod
    def raw_takes(self) -> Iterator[RawTake]:
        """ Yield one RawTake for every usable take, in a stable (sorted) order """

    def split_groups(self, groups, seed, fractions=(0.6, 0.2, 0.2)) -> dict[str, str]:
        """ Default: shuffle groups and cut into train/val/test. Override for dataset rules.

        Returns {group: "train" | "val" | "test"}. Whole groups are assigned, never single takes.
        """
        groups = sorted(set(groups))        # sort first so the shuffle only depends on seed
        random.Random(seed).shuffle(groups)
        n_train = round(fractions[0] * len(groups))
        n_val = round(fractions[1] * len(groups))
        return {g: SPLITS[0] if i < n_train else SPLITS[1] if i < n_train + n_val else SPLITS[2]
                for i, g in enumerate(groups)}




class GuitarSetAdapter(DatasetAdapter):
    def __init__(self, root: Path, audio_subdir="audio_mono-mic_comp",
                 audio_suffix="_mic.wav", labels="instructed"):
        self.root = root
        self.annotation_dir = root / "annotation"
        self.audio_dir =  root / audio_subdir
        self.audio_suffix = audio_suffix
        self.labels = labels        # "instructed" -> first chord annotation, "performed" -> second chord annotation

        # fail early if a folder is missing, not 100 takes into the run
        for d in (self.annotation_dir, self.audio_dir):
            if not d.is_dir():
                raise FileNotFoundError(d)


    def raw_takes(self):
        for p in sorted(self.annotation_dir.glob("*.jams")):
            song_id = p.stem
            # Skip solos - not useful for chord detection
            if "solo" in song_id.split("_"):
                continue

            group = song_id.split("_")[1].split("-")[0]
            audio_path = self.audio_dir / (song_id + self.audio_suffix)
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            anno = ChordAnnotation.from_jams(p, index=0 if self.labels=="instructed" else 1)
            yield RawTake(song_id, group, audio_path, anno)


    def split_groups(self, groups, seed) -> dict[str, str]:
        """ One progression per style in each split: of BN1, BN2, BN3, one goes to train, one to val, one to test """
        rng = random.Random(seed)           # one generator for all styles, so the result only depends on seed
        out = {}
        for style in sorted({g.rstrip("0123456789") for g in groups}):
            progs = sorted({g for g in groups if g.rstrip("0123456789") == style})
            if len(progs) != len(SPLITS):
                raise ValueError(f"style {style!r} has {len(progs)} progressions, expected {len(SPLITS)}")
            rng.shuffle(progs)
            out.update(zip(progs, SPLITS))
        return out




def prepare(adapter: DatasetAdapter, out_dir: Path, seed: int, split_path: Path | None = None) -> list[TakeRecord]:
    """Write .lab files and manifest.csv to out_dir, and the split to split_path (default: out_dir/split.json).

    An existing split file is never silently replaced: if it holds a different split, this raises.
    Changing the official split (splits/<dataset>.json) has to be a deliberate act.
    """
    lab_dir = out_dir / "labs"
    lab_dir.mkdir(parents=True, exist_ok=True)
    takes = []
    for rt in adapter.raw_takes():
        lab = lab_dir / f"{rt.song_id}.lab"
        rt.chords.to_lab(lab)
        takes.append(TakeRecord(rt.song_id, rt.group, rt.audio, lab))
    # manifest: one CSV row per take, paths relative to the manifest so the folder can move
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["song_id", "group", "audio", "lab"])
        w.writeheader()
        for t in takes:
            w.writerow({"song_id": t.song_id, "group": t.group,
                        "audio": os.path.relpath(t.audio, out_dir), "lab": os.path.relpath(t.lab, out_dir)})

    # split: whole groups, decided by the adapter's rule
    group_split = adapter.split_groups({t.group for t in takes}, seed)
    split = {name: [t.song_id for t in takes if group_split[t.group] == name] for name in SPLITS}
    split_path = Path(split_path) if split_path else out_dir / "split.json"
    if split_path.exists() and load_split(split_path) != split:
        raise ValueError(f"{split_path} already holds a different split; delete it on purpose to replace it")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({"seed": seed, **split}, indent=2))
    return takes




def load_manifest(path) -> list[TakeRecord]:
    root = Path(path).parent
    with open(path, newline="") as f:
        return [TakeRecord(r["song_id"], r["group"], root / r["audio"], root / r["lab"])
                for r in csv.DictReader(f)]


def load_split(path) -> dict[str, list[str]]:
    split = json.loads(Path(path).read_text())
    split.pop("seed", None)
    return split