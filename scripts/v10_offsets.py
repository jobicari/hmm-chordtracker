"""V10 step 2: one log offset per quality (maj fixed at 0), added to the posterior before decoding.

Run by `python scripts/run_v10.py offsets`. decode_v10.quality_offsets scales each quality block of the
60 posteriors by exp(offset), then V8's decoders run unchanged (prior power 0.25, Billboard matrix).

Fitting: decode_v10.coordinate_search over OFFSET_GRID for min, 7, maj7, min7, maximising the mean
sevenths WCSR of the fitting sets at one decoder setting:
    lag 8   fixed-lag Viterbi at lag 8, p_self fixed at the value best for those sets without offsets
    network the frame argmax (no decoder)
The lag-8 offsets are then applied at every point (network alone, lag 2, 4, 8), with p_self chosen per
point as V8 does; the network-alone offsets only at the network-alone point.

Two guards against fitting validation noise:
    (a) seeds: fit on seed 0's posteriors (all three sets), judge on seed 1's, and the reverse
    (b) sets: on seed 1, fit on two sets and judge on the third (p_self chosen on the two)
Also: the share of each predicted quality against the reference on RWC-P val, and the offsets fitted on
each seed applied to the step 1 ensemble.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import run_v10 as r
from chordtracker import decode_v10, online_hmm, transitions_v8

OFFSET_GRID = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
FREE = [1, 2, 3, 4]                 # min, 7, maj7, min7; maj stays 0
ZERO = np.zeros(len(r.QUALITIES))

_W: dict = {}


def _init(posts, A, prior):
    _W.update(posts=posts, A=A, prior=prior)


def _lag8(job):
    key, offsets = job
    post = decode_v10.quality_offsets(_W["posts"][key], offsets)
    return online_hmm.fixed_lag_viterbi(post, _W["A"], _W["prior"], [8])[8]


def objective_lag8(pool, keys, songs, sets):
    def f(offsets):
        paths = pool.map(_lag8, [(k, offsets) for k in keys], chunksize=4)
        sc = r.pooled({k: {("lag 8", 0): p} for k, p in zip(keys, paths)}, songs, sets=sets)
        return sc[("mean", "lag 8", 0)]["sevenths"]
    return f


def objective_network(posts, keys, songs, sets):
    def f(offsets):
        paths = {k: {("network", None): decode_v10.quality_offsets(posts[k], offsets).argmax(axis=0)} for k in keys}
        return r.pooled(paths, songs, sets=sets)[("mean", "network", None)]["sevenths"]
    return f


def fit(cfg, songs, posts, prior, sets, label):
    """{lag 8, network}: (offsets, objective, start objective, p_self) fitted on `sets` of these posteriors."""
    keys = [k for k in posts if songs[k]["set"] in sets]
    base = r.pooled(r.decode_all({k: posts[k] for k in keys}, cfg, prior, 0.25, lags=[8]), songs, sets=sets)
    p_self = r.best_points({k: v for k, v in base.items() if k[1] == "lag 8"})["lag 8"]
    A = r.transition_mats(cfg, prior, [p_self])[p_self]
    out = {}
    with ProcessPoolExecutor(8, initializer=_init,
                             initargs=({k: posts[k] for k in keys}, A, transitions_v8.tempered_prior(prior, 0.25))) as pool:
        x, best, trace = decode_v10.coordinate_search(objective_lag8(pool, keys, songs, sets), ZERO, OFFSET_GRID, FREE)
    out["lag 8"] = {"offsets": x.tolist(), "objective": best, "start": trace[0]["objective"], "p_self": p_self,
                    "evaluations": len(trace)}
    x, best, trace = decode_v10.coordinate_search(objective_network(posts, keys, songs, sets), ZERO, OFFSET_GRID, FREE)
    out["network"] = {"offsets": x.tolist(), "objective": best, "start": trace[0]["objective"], "evaluations": len(trace)}
    print(f"fit {label} on {sets}: " + "; ".join(
        f"{k} {np.round(v['offsets'], 2).tolist()} {100 * v['start']:.2f} -> {100 * v['objective']:.2f}"
        for k, v in out.items()), flush=True)
    return out


def with_offsets(posts, offsets):
    return {k: decode_v10.quality_offsets(p, np.asarray(offsets)) for k, p in posts.items()}


def judged(cfg, songs, posts, prior, fitted):
    """Point table of these posteriors with the lag-8 offsets at every point, the network offsets at the network."""
    sc = r.pooled(r.decode_all(with_offsets(posts, fitted["lag 8"]["offsets"]), cfg, prior, 0.25), songs)
    net = r.pooled({k: {("network", None): p.argmax(axis=0)}
                    for k, p in with_offsets(posts, fitted["network"]["offsets"]).items()}, songs)
    rows = r.point_table(sc)
    return rows, r.point_table(net), sc


def shares(posts, prior, cfg, songs, offsets, dataset="rwc"):
    """Per quality: reference share and predicted share (network alone, lag 8 at V8's p_self) on frames with a target."""
    keys = [k for k in posts if songs[k]["set"] == dataset]
    p = with_offsets({k: posts[k] for k in keys}, offsets)
    paths = r.decode_all(p, cfg, prior, 0.25, p_selfs=[0.95], lags=[8])
    count = {name: np.zeros(len(r.QUALITIES)) for name in ("reference", "network", "lag 8")}
    for k in keys:
        y = songs[k]["targets"]
        keep = y >= 0
        count["reference"] += np.bincount(y[keep] // 12, minlength=len(r.QUALITIES))
        count["network"] += np.bincount(paths[k][("network", None)][keep] // 12, minlength=len(r.QUALITIES))
        count["lag 8"] += np.bincount(paths[k][("lag 8", 0.95)][keep] // 12, minlength=len(r.QUALITIES))
    return {name: (100 * c / c.sum()).tolist() for name, c in count.items()}


def share_table(rows: dict[str, dict]) -> str:
    lines = ["| RWC-P val, share of frames % | " + " | ".join(r.QUALITIES) + " |",
             "|---|" + "---:|" * len(r.QUALITIES)]
    done_ref = False
    for name, s in rows.items():
        if not done_ref:
            lines.append("| reference | " + " | ".join(f"{v:.1f}" for v in s["reference"]) + " |")
            done_ref = True
        for point in ("network", "lag 8"):
            lines.append(f"| {name}, {point} | " + " | ".join(f"{v:.1f}" for v in s[point]) + " |")
    return "\n".join(lines)


def run(cfg, songs):
    posts = {s: r.load_posteriors(r.SEEDS[s]) for s in r.SEEDS}
    priors = {s: r.prior_of(r.SEEDS[s]) for s in r.SEEDS}
    text, data = [], {}

    # (a) fit on one seed (all sets), judge on the other
    fits = {s: fit(cfg, songs, posts[s], priors[s], r.SETS, f"seed {s}") for s in r.SEEDS}
    data["fits"] = fits
    tables = {}
    for s in r.SEEDS:
        tables[f"seed {s}, V8"] = r.point_table(r.pooled(r.decode_all(posts[s], cfg, priors[s], 0.25), songs))
        for f in r.SEEDS:
            rows, net, _ = judged(cfg, songs, posts[s], priors[s], fits[f])
            tables[f"seed {s}, offsets fitted on seed {f}"] = rows
            tables[f"seed {s}, network offsets fitted on seed {f}"] = net
    text.append("Guard (a), seeds. Offsets fitted on one seed (three sets), applied to both seeds.\n\n"
                + offsets_table({f"seed {s}": fits[s] for s in r.SEEDS}) + "\n\n" + r.markdown("posteriors", tables))
    print(text[-1], flush=True)

    # (b) leave one set out, seed 1
    loso, ltables = {}, {}
    for held in r.SETS:
        fitting = tuple(d for d in r.SETS if d != held)
        f = fit(cfg, songs, posts[1], priors[1], fitting, f"seed 1 without {held}")
        loso[held] = f
        keys = [k for k in posts[1] if songs[k]["set"] == held]
        p_self = f["lag 8"]["p_self"]
        for name, offsets in (("V8", ZERO), ("offsets", f["lag 8"]["offsets"])):
            sc = r.pooled(r.decode_all(with_offsets({k: posts[1][k] for k in keys}, offsets), cfg, priors[1], 0.25),
                          songs, sets=(held,))
            chosen = {"network": None, "lag 2": None, "lag 4": None, "lag 8": p_self}
            fit_sc = r.pooled(r.decode_all(with_offsets({k: posts[1][k] for k in posts[1] if songs[k]["set"] in fitting},
                                                        offsets), cfg, priors[1], 0.25), songs, sets=fitting)
            chosen.update({pt: p for pt, p in r.best_points(fit_sc).items() if pt != "network"})
            ltables[(held, name)] = {pt: sc[(held, pt, chosen[pt])] for pt in r.POINT_ORDER}
        net = {k: {("network", None): decode_v10.quality_offsets(posts[1][k], np.asarray(f["network"]["offsets"]))
                   .argmax(axis=0)} for k in keys}
        ltables[(held, "network offsets")] = {"network": r.pooled(net, songs, sets=(held,))[(held, "network", None)]}
    data["leave_one_set_out"] = loso
    text.append("Guard (b), sets (seed 1). Fitted on the two other sets; p_self per point chosen on those two.\n\n"
                + offsets_table({f"without {h}": loso[h] for h in r.SETS}) + "\n\n" + loso_table(ltables))
    print(text[-1], flush=True)

    # quality shares on RWC-P val, seed 0
    sh = {"seed 0, V8": shares(posts[0], priors[0], cfg, songs, ZERO),
          "seed 0, offsets (seed 0 fit)": shares(posts[0], priors[0], cfg, songs, fits[0]["lag 8"]["offsets"])}
    data["shares"] = sh
    text.append("Quality shares, RWC-P val, seed 0 (lag 8 at p_self 0.95), frames with a reference target.\n\n"
                + share_table(sh))
    print(text[-1], flush=True)

    # offsets on the ensemble
    prior = (priors[0] + priors[1]) / 2
    etables = {}
    for mode in ("geometric", "arithmetic"):
        comb = {k: decode_v10.ensemble([posts[0][k], posts[1][k]], mode) for k in posts[0]}
        etables[f"{mode} ensemble"] = r.point_table(r.pooled(r.decode_all(comb, cfg, prior, 0.25), songs))
        for f in r.SEEDS:
            rows, net, _ = judged(cfg, songs, comb, prior, fits[f])
            etables[f"{mode} ensemble, offsets fitted on seed {f}"] = rows
    text.append("Offsets (fitted on one seed, not on the ensemble) applied to the ensembles, prior power 0.25.\n\n"
                + r.markdown("posteriors", etables))
    r.write("step2_offsets", "\n\n".join(text), data)


def offsets_table(fits: dict[str, dict]) -> str:
    lines = ["| fitted on | objective | p_self | " + " | ".join(r.QUALITIES) + " | mean sevenths before -> after |",
             "|---|---|---:|" + "---:|" * len(r.QUALITIES) + "---|"]
    for name, f in fits.items():
        for obj in ("lag 8", "network"):
            v = f[obj]
            p = f"{v['p_self']:g}" if "p_self" in v else ""
            lines.append(f"| {name} | {obj} | {p} | " + " | ".join(f"{o:+.2f}" for o in v["offsets"])
                         + f" | {100 * v['start']:.2f} -> {100 * v['objective']:.2f} |")
    return "\n".join(lines)


def loso_table(t: dict) -> str:
    lines = ["| held-out set | system | " + " | ".join(r.POINT_NAMES[p] for p in r.POINT_ORDER) + " |",
             "|---|---|" + "---:|" * len(r.POINT_ORDER)]
    for (held, name), by in t.items():
        cells = [f"{100 * by[p]['majmin']:.2f} / {100 * by[p]['sevenths']:.2f}" if p in by else "" for p in r.POINT_ORDER]
        lines.append(f"| {held} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
