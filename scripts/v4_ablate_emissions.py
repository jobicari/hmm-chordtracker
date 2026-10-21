"""V4 ablation 1 (val): learned emissions on the V2 beat features, against the V2 template emission.

Fit on TRAIN beats only (12 transpositions of every beat), choose C / p_state / p_self on val.
Every model sees the same 24 numbers per beat: the V2 treble and bass beat-mean chroma.

    PYTHONPATH=src python scripts/v4_ablate_emissions.py <beats.pkl> <out.json>
"""
import json, pickle, re, sys
import librosa, numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from chordtracker import bass, beats, evaluate, templates as tpl, transitions as T
from chordtracker.emission import TemplateEmission

cache = pickle.load(open(sys.argv[1], "rb"))
LABELS24 = tpl.majmin_labels()[:24]


def beat_targets(take):
    """Class index (root + 12 * is_minor) of the reference chord covering most of each beat, -1 if none."""
    ref, edges = take["reference"], take["edges"]
    out = np.full(len(edges) - 1, -1)
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        best, best_overlap = None, 0.0
        for (s, e), lab in ref:
            overlap = min(b, e) - max(a, s)
            if overlap > best_overlap:
                best, best_overlap = lab, overlap
        c = T.chord_class(best) if best else None
        if c is not None:
            out[k] = c[0] + 12 * (c[1] == "min")
    return out


def rotate(X, y, shift):
    """Transpose beat features (24, n) and classes by shift semitones."""
    Xr = np.concatenate([np.roll(X[:12], shift, axis=0), np.roll(X[12:], shift, axis=0)])
    yr = np.where(y >= 0, (y % 12 + shift) % 12 + 12 * (y // 12), -1)
    return Xr, yr


def features(take):
    return np.concatenate([take["treble"], take["bass"]])


X_tr, y_tr = [], []
for take in cache["train"]:
    X, y = features(take), beat_targets(take)
    for s in range(12):
        Xr, yr = rotate(X, y, s)
        X_tr.append(Xr[:, yr >= 0].T); y_tr.append(yr[yr >= 0])
X_tr, y_tr = np.concatenate(X_tr), np.concatenate(y_tr)
prior = np.bincount(y_tr, minlength=24) / len(y_tr)
print(f"train beats x12: {len(y_tr)}, minor share {prior[12:].sum():.3f}", flush=True)

val = cache["val"]
val_y = [beat_targets(t) for t in val]


def template_post(take):
    model = TemplateEmission(tpl.majmin_templates(0.25), tpl.majmin_labels(), beta=1.0)
    return bass.posteriors(model, take["treble"], take["bass"], 0.2)


def evid(take, post):
    return beats.BeatEvidence(take["song_id"], post, take["edges"], take["reference"], take["duration"])


def decode_all(posts, labels, p_self, p_state=None):
    n = posts[0].shape[0]
    A = librosa.sequence.transition_loop(n, p_self)
    scores, cpm, per = [], [], {}
    for take, P in zip(val, posts):
        path = librosa.sequence.viterbi_discriminative(P, A, p_state=p_state)
        sc, ch = beats.score_path(path, labels, evid(take, P))
        scores.append(sc); cpm.append(ch); per[take["song_id"]] = sc.wcsr
    return evaluate.pooled(scores).wcsr, float(np.mean(cpm)), per


def beat_stats(posts):
    """Val beat accuracy (argmax vs majority label) and mean log-loss of the true class (calibration)."""
    hit, nll, n = 0, 0.0, 0
    for P, y in zip(posts, val_y):
        m = y >= 0
        hit += int((P[:24].argmax(0)[m] == y[m]).sum())
        nll += float(-np.log(np.clip(P[y[m], np.flatnonzero(m)], 1e-12, None)).sum()); n += int(m.sum())
    return hit / n, nll / n


P_GRID = [0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
rows, per_take = [], {}


def run(name, posts, labels, p_state=None, extra=None):
    acc, nll = beat_stats(posts)
    best = None
    for p in P_GRID:
        w, c, per = decode_all(posts, labels, p, p_state)
        row = {"model": name, "p_self": p, "wcsr": w, "cpm": c, "beat_acc": acc, "nll": nll, **(extra or {})}
        rows.append(row)
        if best is None or w > best[0]["wcsr"]:
            best = (row, per)
    per_take[name] = best[1]
    r = best[0]
    print(f"{name:34s} beat acc {100*acc:5.1f}%  nll {nll:5.2f}  best p {r['p_self']:.2f}  "
          f"{100*r['wcsr']:6.2f}%  {r['cpm']:5.1f}/min", flush=True)


run("template (V2)", [template_post(t) for t in val], tpl.majmin_labels())

for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    lr = LogisticRegression(C=C, max_iter=3000).fit(X_tr, y_tr)
    posts = [lr.predict_proba(features(t).T).T for t in val]
    run(f"logreg C={C:g}", posts, LABELS24, extra={"C": C})
    run(f"logreg C={C:g} / train prior", posts, LABELS24, p_state=prior, extra={"C": C})

for name, est in [("gaussian shared cov (LDA)", LinearDiscriminantAnalysis()),
                  ("gaussian per chord (QDA)", QuadraticDiscriminantAnalysis(reg_param=0.1))]:
    est.fit(X_tr, y_tr)
    posts = [est.predict_proba(features(t).T).T for t in val]
    run(name, posts, LABELS24)
    run(name + " / train prior", posts, LABELS24, p_state=prior)

json.dump({"train_prior": prior.tolist(), "rows": rows, "per_take": per_take}, open(sys.argv[2], "w"))
