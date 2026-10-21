"""V4 ablation 2 (val): the V3 change priors (train / Billboard counts, tau) at full strength on the
two calibrated emissions of ablation 1: logistic regression C 10 with p_state = train prior, and the
templates with beta fitted on train. Counts from train labels or Billboard only; val-fit is a ceiling.

    PYTHONPATH=src python scripts/v4_ablate_priors.py <beats.pkl> <out.json>
"""
import json, sys
import numpy as np
from scipy.optimize import minimize_scalar
sys_argv = list(sys.argv)
sys.argv = [sys_argv[0], sys_argv[1], "/dev/null"]
exec(open(__file__.replace("ablate_priors", "ablate_emissions")).read().split('run("template (V2)"')[0])
sys.argv = sys_argv
from chordtracker import paths
from chordtracker.data.data import ChordAnnotation, load_split

# emissions
lr = LogisticRegression(C=10, max_iter=3000).fit(X_tr, y_tr)
post_lr = [lr.predict_proba(features(t).T).T for t in val]
tm = TemplateEmission(tpl.majmin_templates(0.25), tpl.majmin_labels(), beta=1.0)
roots = bass.root_matrix(0.25)
score = lambda tr, ba: (tm.scores(tr) + 0.2 * roots @ ba)[:24]
S_tr = score(X_tr[:, :12].T, X_tr[:, 12:].T)
def nll(b):
    z = b * S_tr; z = z - z.max(0); lp = z - np.log(np.exp(z).sum(0)); return -lp[y_tr, np.arange(len(y_tr))].mean()
beta = minimize_scalar(nll, bounds=(0.1, 50), method="bounded").x
def soft(s):
    s = s - s.max(0); e = np.exp(s); return e / e.sum(0)
post_tb = [soft(beta * score(t["treble"], t["bass"])) for t in val]
post_v2 = [soft(1.0 * score(t["treble"], t["bass"])) for t in val]
EMISSIONS = {"logreg C10 / train prior": (post_lr, prior), f"templates beta {beta:.2f}": (post_tb, None),
             "templates beta 1 (V2)": (post_v2, None)}

# priors (24 states: drop N from the V3 Q and renormalize)
split = load_split(paths.SPLITS_DIR / "guitarset.json")
labs = paths.DATA_DIR / "processed" / "guitarset" / "labs"
seqs = lambda ids: [ChordAnnotation.from_lab(labs / f"{s}.lab").labels for s in ids]
counts = {"train": T.change_counts(seqs(split["train"])),
          "billboard": np.array(json.load(open(paths.REPO_ROOT / "results/v3/billboard_counts.json"))["counts"], float),
          "val-fit": T.change_counts(seqs(split["val"]))}
def Q24(c, tau):
    Q = T.change_matrix(c, alpha=1.0, tau=tau)[:24, :24]; return Q / Q.sum(1, keepdims=True)
TAU = [0.0, 0.25, 0.5, 1.0, 1.5]

rows, per_take = [], {}
for ename, (posts, p_state) in EMISSIONS.items():
    for source in counts:
        for tau in TAU:
            if tau == 0.0 and source != "train":
                continue
            Q, best = Q24(counts[source], tau), None
            for p in P_GRID:
                A = p * np.eye(24) + (1 - p) * Q
                sc, cpm, per = [], [], {}
                for take, P in zip(val, posts):
                    path = librosa.sequence.viterbi_discriminative(P, A, p_state=p_state)
                    s, c = beats.score_path(path, LABELS24, evid(take, P))
                    sc.append(s); cpm.append(c); per[take["song_id"]] = s.wcsr
                row = {"emission": ename, "source": "uniform" if tau == 0 else source, "tau": tau, "p_self": p,
                       "wcsr": evaluate.pooled(sc).wcsr, "cpm": float(np.mean(cpm))}
                rows.append(row)
                if best is None or row["wcsr"] > best[0]["wcsr"]:
                    best = (row, per)
            per_take[f"{ename} | {best[0]['source']} tau {tau}"] = best[1]
            r = best[0]
            print(f"{ename:26s} {r['source']:9s} tau {tau:4.2f}  p {r['p_self']:.2f}  {100*r['wcsr']:6.2f}%  {r['cpm']:5.1f}/min", flush=True)
json.dump({"beta": beta, "rows": rows, "per_take": per_take}, open(sys.argv[2], "w"))
