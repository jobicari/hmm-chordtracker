"""V3 ablation 1 (val): learned chord-change prior Q from the train labels, p_self x tau grid.
Decodes the cached V2 val evidence (cache_val.py). Q is counted on TRAIN lead-sheet labels only;
the "val-fit" rows count Q on the val labels themselves, a ceiling for what any change prior could
add, never a candidate.

    PYTHONPATH=src python scripts/v3_ablate_transitions.py <v2_val_evidence.pkl> <out.json> [results/v3/billboard_counts.json]
"""
import json, pickle, sys
import librosa, numpy as np
from chordtracker import beats, evaluate, paths, transitions as T
from chordtracker.data.data import ChordAnnotation, load_split

cache = pickle.load(open(sys.argv[1], "rb"))
labels, takes = cache["labels"], cache["takes"]
split = load_split(paths.SPLITS_DIR / "guitarset.json")
labs = paths.DATA_DIR / "processed" / "guitarset" / "labs"
seqs = lambda ids: [ChordAnnotation.from_lab(labs / f"{s}.lab").labels for s in ids]
counts = {"train": T.change_counts(seqs(split["train"])), "val-fit": T.change_counts(seqs(split["val"]))}
if len(sys.argv) > 3:      # Billboard counts from count_billboard.py (labels only, no overlap with GuitarSet)
    counts["billboard"] = np.array(json.load(open(sys.argv[3]))["counts"], dtype=float)

P_GRID = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
TAU = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

def run(A):
    scores, changes, per, n_beats = [], [], {}, []
    for ev in takes:
        path = librosa.sequence.viterbi_discriminative(ev.posteriors, A)
        sc, ch = beats.score_path(path, labels, ev)
        scores.append(sc); changes.append(ch); per[ev.song_id] = sc.wcsr
        n_beats.append(np.mean(path == T.N_STATE))
    tot = evaluate.pooled(scores)
    return {"wcsr": tot.wcsr, "cpm": float(np.mean(changes)), "n_share": float(np.mean(n_beats)), "per": per}

rows = []
for source in counts:
    for tau in TAU:
        if source == "val-fit" and tau == 0.0:
            continue
        Q = T.change_matrix(counts[source], alpha=1.0, tau=tau)
        for p in P_GRID:
            r = run(T.transition(Q, p))
            rows.append({"source": source, "tau": tau, "p_self": p, **r})
            print(f"{source:8s} tau {tau:4.2f} p {p:4.2f}  {100*r['wcsr']:6.2f}%  {r['cpm']:5.1f}/min", flush=True)

ref_cpm = float(np.mean([evaluate.changes_per_minute(ev.reference, ev.duration) for ev in takes]))
json.dump({"counts": {k: v.tolist() for k, v in counts.items()}, "reference_changes_per_min": ref_cpm, "rows": rows},
          open(sys.argv[2], "w"))
print("reference changes/min", ref_cpm)
