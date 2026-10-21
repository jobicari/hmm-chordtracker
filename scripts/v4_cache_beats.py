"""Cache the V2 beat features (treble and bass beat-mean chroma), beat edges and reference of every
train and val take, so the V4 emission ablations fit and decode cached numbers. Test is not touched.

    NUMBA_CACHE_DIR=<fresh dir> PYTHONPATH=src python scripts/v4_cache_beats.py <out.pkl>
"""
import pickle, sys
import librosa, numpy as np, yaml
from chordtracker import beats, features_v2, paths
from chordtracker.data.data import ChordAnnotation, load_manifest, load_split

cfg = yaml.safe_load((paths.REPO_ROOT / "configs" / "v2.yaml").read_text())
split = load_split(paths.SPLITS_DIR / "guitarset.json")
records = load_manifest(paths.DATA_DIR / "processed" / "guitarset" / "manifest.csv")
out = {}
for name in ("train", "val"):
    ids = set(split[name])
    takes = []
    for r in [r for r in records if r.song_id in ids]:     # serial: forked workers crash in beat_track (numba)
        treble, bass = features_v2.extract(r.audio, cfg)
        bounds = beats.segment_bounds(beats.beat_frames(r.audio, cfg), treble.shape[1])
        reference = ChordAnnotation.from_lab(r.lab)
        duration = float(reference.intervals[-1, 1])
        edges = librosa.frames_to_time(bounds, sr=cfg["audio"]["sr"], hop_length=cfg["features"]["hop_length"])
        edges[0], edges[-1] = 0.0, duration
        takes.append({"song_id": r.song_id, "treble": beats.beat_sync(treble, bounds, "mean"),
                      "bass": beats.beat_sync(bass, bounds, "mean"), "edges": edges,
                      "reference": reference, "duration": duration})
        print(name, r.song_id, takes[-1]["treble"].shape, flush=True)
    out[name] = takes
pickle.dump(out, open(sys.argv[1], "wb"))
