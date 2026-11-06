"""Component 29 (V8): the curated ChoCo export as songs for training and validation.

The export (choco_corpus_prep, `export/<partition>/`) holds one WAV, one JAMS chord annotation in that
recording's time, and one manifest row per recording. It stays where it is, outside this repository:
cfg["data"]["choco_export"] points at it, and data/choco/SOURCE.txt records its repository and commit.

    choco_takes(split, cfg)          the recordings of 'train' or 'val' (splits/choco.json) -> list[TunedTake]
    check_audio(take)                the WAV is the file the manifest names (its SHA-256)
    tuned_log_cqt_of_audio(y, cfg, tuning_cents)   causal log-CQT at a known tuning, from frame 0
    load(take, cfg)                  -> Song on the cached log-CQT, targets as data_v5.frame_targets

Tuning: each manifest row gives the recording's deviation from A440 in cents, settled together with
its labels (a recording 42 cents flat in the written key, not 58 sharp a semitone lower). The CQT is
computed at that tuning from the first frame, rather than estimated on the first 10 s as the live
path does (features_v7_1.start_tuning): estimating again could land a recording near +/-50 cents a
semitone away from its labels, and A440 before the estimate would put up to half a semitone of error
into the first 10 s of 26 recordings that sit more than 25 cents off.
"""
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from chordtracker import paths
from chordtracker.acoustic_hmm import frame_times
from chordtracker.data.data import ChordAnnotation, load_split
from chordtracker.data_v5 import Song, Take, frame_targets
from chordtracker.features_v7_1 import causal_harmonic, delayed_cqt, running_level

CHOCO_SPLIT = paths.SPLITS_DIR / "choco.json"
PARTITIONS = ("isophonics", "uspop2002", "robbie-williams", "jaah")


@dataclass(frozen=True)
class TunedTake(Take):
    """A take whose tuning is known: cents from A440, and the SHA-256 of its WAV."""
    tuning_cents: float
    sha256: str


def export_dir(cfg: dict) -> Path:
    """The export directory named in cfg["data"]["choco_export"] (~ expanded)."""
    return Path(cfg["data"]["choco_export"]).expanduser()


def manifest(cfg: dict) -> dict[str, dict]:
    """Every manifest row of the export, by recording id (e.g. 'isophonics_0')."""
    rows = {}
    for partition in PARTITIONS:
        with open(export_dir(cfg) / partition / "manifest.csv", newline="") as f:
            rows.update({r["id"]: r for r in csv.DictReader(f)})
    return rows


def choco_takes(split: str, cfg: dict, split_file: Path = CHOCO_SPLIT) -> list[TunedTake]:
    """The ChoCo recordings of 'train' or 'val', with their annotation and tuning."""
    if split == "test":
        raise ValueError("ChoCo test is evaluated once, by the frozen run script, never loaded here")
    rows = manifest(cfg)
    takes = []
    for rec in load_split(split_file)[split]:
        r = rows[rec]
        folder = export_dir(cfg) / r["partition"]
        takes.append(TunedTake(f"choco/{rec}", folder / r["audio"], ChordAnnotation.from_jams(folder / r["jams"]),
                               float(r["tuning_cents"]), r["sha256"]))
    return takes


def check_audio(take: TunedTake) -> None:
    """Raise if the WAV is missing or is not the file the manifest names."""
    if not take.audio.exists():
        raise FileNotFoundError(f"{take.key}: {take.audio} missing (choco-prep partitions export-audio)")
    digest = hashlib.sha256()
    with open(take.audio, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != take.sha256:
        raise ValueError(f"{take.key}: {take.audio} is not the recording in the manifest (SHA-256 differs)")


def tuning_bins(tuning_cents: float, bins_per_octave: int) -> float:
    """Cents -> fractions of a CQT bin, as librosa's `tuning` takes it: 24 bins/octave -> 50 cents a bin."""
    return tuning_cents * bins_per_octave / 1200.0


def tuned_log_cqt_of_audio(y: np.ndarray, cfg: dict, tuning_cents: float) -> np.ndarray:
    """features_v7_1.causal_log_cqt_of_audio with the tuning given and applied from the first frame."""
    q, comp = cfg["cqt"], cfg["compression"]
    y_h = causal_harmonic(y, cfg["audio"]["sr"], cfg)
    C = delayed_cqt(y_h, cfg, tuning_bins(tuning_cents, q["bins_per_octave"]))
    level = running_level(C, comp["level_percentile"])
    return np.log1p(comp["gamma"] * C / level[None, :]).astype(np.float32)


def cache_dir(cfg: dict) -> Path:
    """data/processed/v8_logcqt_<fmin>/ (the key adds choco/<id>.npy)."""
    return paths.DATA_DIR / "processed" / f"v8_logcqt_{cfg['cqt']['fmin'].replace('#', 's')}"


def load(take: TunedTake, cfg: dict) -> Song:
    """Cached tuned log-CQT of the take on cfg's grid, plus the target at the centre time of every frame."""
    file = cache_dir(cfg) / f"{take.key}.npy"
    if file.exists():
        cqt = np.load(file).astype(np.float32)
    else:
        y, _ = librosa.load(str(take.audio), sr=cfg["audio"]["sr"], mono=cfg["audio"]["mono"])
        cqt = tuned_log_cqt_of_audio(y, cfg, take.tuning_cents)
        file.parent.mkdir(parents=True, exist_ok=True)
        np.save(file, cqt.astype(np.float16))
    times = frame_times(cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    return Song(take.key, cqt, frame_targets(take.annotation, times))
