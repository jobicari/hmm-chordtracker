"""V3 step 3 (val): k sub-states per chord = minimum duration k beats, p_self re-swept.
Chord c becomes the chain c_1 -> c_2 -> ... -> c_k; c_1..c_{k-1} move on with probability 1,
c_k stays with p_self or leaves to the first sub-state of another chord with (1 - p_self) Q.
Each sub-state gets its chord's posterior / k (columns still sum to 1; the uniform marginal is
unchanged), and the decoded path is mapped back to chords. k = 1 is the plain V2/V3 decoder.

    PYTHONPATH=src python scripts/v3_ablate_min_duration.py <v2_val_evidence.pkl> <out.json> [results/v3/billboard_counts.json]
"""
import json, pickle, sys
import librosa, numpy as np
from chordtracker import beats, evaluate, transitions as T

cache = pickle.load(open(sys.argv[1], "rb"))
labels, takes = cache["labels"], cache["takes"]
priors = {"uniform": T.change_matrix(np.zeros((2, 2, 12)), tau=0.0)}
if len(sys.argv) > 3:
    priors["billboard tau 0.25"] = T.change_matrix(np.array(json.load(open(sys.argv[3]))["counts"], dtype=float), alpha=1.0, tau=0.25)

def expanded(Q, p_self, k):
    n = len(Q); A = np.zeros((n * k, n * k))
    for c in range(n):
        for s in range(k - 1):
            A[c * k + s, c * k + s + 1] = 1.0
        last = c * k + k - 1
        A[last, last] = p_self
        for d in range(n):
            if d != c:
                A[last, d * k] = (1 - p_self) * Q[c, d]
    return A

rows = []
for name, Q in priors.items():
    for k in [1, 2, 3, 4]:
        for p in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
            A = expanded(Q, p, k); scores, changes, per = [], [], {}
            for ev in takes:
                path = librosa.sequence.viterbi_discriminative(np.repeat(ev.posteriors, k, axis=0) / k, A) // k
                sc, ch = beats.score_path(path, labels, ev)
                scores.append(sc); changes.append(ch); per[ev.song_id] = sc.wcsr
            w = evaluate.pooled(scores).wcsr
            rows.append({"prior": name, "k": k, "p_self": p, "wcsr": w, "cpm": float(np.mean(changes)), "per": per})
            print(f"{name:20s} k {k} p {p:4.2f}  {100*w:6.2f}%  {np.mean(changes):5.1f}/min", flush=True)
json.dump(rows, open(sys.argv[2], "w"))
