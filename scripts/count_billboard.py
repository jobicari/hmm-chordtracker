"""Transposition-invariant chord-change counts from the McGill Billboard annotations (ChoCo v1.0.0
JAMS, partitions/billboard/choco/jams, zenodo.org/records/7706751). Labels only: no audio, and no
song in common with GuitarSet. Duplicate songs (same title and artist) are counted once.

    PYTHONPATH=src python scripts/count_billboard.py <jams_dir> <out.json>    # (2, 2, 12) counts
"""
import glob, json, sys
import numpy as np
from chordtracker import transitions as T

songs = {}
for path in sorted(glob.glob(f"{sys.argv[1]}/*.jams")):
    d = json.load(open(path))
    key = (d["file_metadata"].get("title"), d["file_metadata"].get("artist"))
    chords = [a for a in d["annotations"] if a["namespace"] == "chord"]
    if key not in songs and chords:
        songs[key] = [o["value"] for o in chords[0]["data"]]
counts = T.change_counts(list(songs.values()))
json.dump({"source": f"McGill Billboard via ChoCo v1.0.0 (zenodo.org/records/7706751), partitions/billboard/choco/jams; "
                     f"{len(songs)} songs after de-duplication",
           "axes": ["quality_from (maj, min)", "quality_to (maj, min)", "root interval in semitones (0..11)"],
           "counts": counts.astype(int).tolist()}, open(sys.argv[2], "w"), indent=1)
print(f"{len(songs)} songs, {int(counts.sum())} changes")
