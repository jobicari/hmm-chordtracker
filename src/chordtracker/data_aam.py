"""Component 14b (V5.2): AAM songs as training takes for the V5 network.

AAM (Artificial Audio Multitracks, Zenodo 5794629, v1.1.0, CC BY 4.0) is generated from music-theory
rules and rendered with real instrument samples, so its chords and beats are exact. Only the log-CQT
of each mix is kept (scripts/fetch_aam.py); the FLAC files are gone, so a take's audio path is never
opened: data_v5.load reads the cache by the take's key ("aam/0001").

    harte(label)            AAM chord name -> Harte label: 'A#maj' -> 'A#:maj', 'N.C.' -> 'N', else 'X'
    aam_annotation(song)    <song>_beatinfo.arff (+ the 'end' mark of _segments.arff) -> ChordAnnotation
    aam_song_ids()          the songs whose cache entry is complete (data/aam/fetched.txt)
    aam_takes(song_ids)     -> list[data_v5.Take], loaded with data_v5.load like RWC-P and GuitarSet

Labels come per beat: a beat lasts until the next one starts, and the last beat lasts until the
song's 'end' mark in <song>_segments.arff (the median beat length if there is none). Frames after the last beat (the mix's reverb tail, about 2 to 3 s) and N.C. or
BASS_NOTE_EXCEPTION beats get no target, as N and X do in V5.
"""
import re

import numpy as np

from chordtracker import paths
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5 import Take

AAM_DIR = paths.DATA_DIR / "aam"
CHORD = re.compile(r"^([A-G][#b]?)(maj|min)$")


def harte(label: str) -> str:
    """AAM chord name -> Harte label. Examples: 'Fmaj' -> 'F:maj', 'A#min' -> 'A#:min', 'N.C.' -> 'N'."""
    if label == "N.C.":
        return "N"
    match = CHORD.match(label)
    return f"{match[1]}:{match[2]}" if match else "X"


def aam_annotation(song_id: str) -> ChordAnnotation:
    """Beat start times and chord names of one song -> one segment per beat."""
    file = AAM_DIR / "annotations" / f"{song_id}_beatinfo.arff"
    rows = [line.rsplit(",", 3) for line in file.read_text().splitlines()
            if line.strip() and not line.startswith("@")]
    starts = np.array([float(start) for start, _, _, _ in rows])
    labels = [harte(chord.strip().strip("'")) for _, _, _, chord in rows]
    ends = np.append(starts[1:], song_end(song_id) or starts[-1] + np.median(np.diff(starts)))
    return ChordAnnotation(np.stack([starts, ends], axis=1), labels)


def song_end(song_id: str) -> float | None:
    """Time of the 'end' mark in <song>_segments.arff, or None when the file or the mark is missing."""
    file = AAM_DIR / "annotations" / f"{song_id}_segments.arff"
    if not file.exists():
        return None
    for line in file.read_text().splitlines():
        fields = line.split(",")
        if len(fields) > 1 and fields[1].strip().strip("'") == "end":
            return float(fields[0])
    return None


def aam_song_ids() -> list[str]:
    """Song ids ('0001' ...) listed in fetched.txt, i.e. cached completely, sorted."""
    return sorted(set((AAM_DIR / "fetched.txt").read_text().split()))


def aam_takes(song_ids: list[str]) -> list[Take]:
    """Takes keyed 'aam/<id>', read from the log-CQT cache; the audio path is a placeholder."""
    return [Take(f"aam/{s}", AAM_DIR / "mixes" / f"{s}_mix.flac", aam_annotation(s)) for s in song_ids]
