"""Cache the V2 beat evidence (posteriors, beat edges, reference) of every val take, so the V3
transition ablations decode cached posteriors. V3 does not change the emission. Val only.

    NUMBA_CACHE_DIR=<fresh dir> PYTHONPATH=src python scripts/v3_cache_val.py <out.pkl>
"""
import pickle, sys
from pathlib import Path
import yaml
from chordtracker import bass, emission, paths
from chordtracker.data.data import load_manifest, load_split

cfg = yaml.safe_load((paths.REPO_ROOT / "configs" / "v2.yaml").read_text())
model = emission.TemplateEmission.from_config(cfg)
val = set(load_split(paths.SPLITS_DIR / "guitarset.json")["val"])
records = [r for r in load_manifest(paths.DATA_DIR / "processed" / "guitarset" / "manifest.csv") if r.song_id in val]
takes = []
for r in records:             # serial: forked workers crash in beat_track (numba)
    takes.append(bass.take_evidence(r, model, cfg))
    print(r.song_id, takes[-1].posteriors.shape, flush=True)
pickle.dump({"labels": model.labels, "takes": takes}, open(sys.argv[1], "wb"))
