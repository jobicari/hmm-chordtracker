"""Run V14: V8's causal CRNN with an N (no chord) class, trained, decoded and scored on validation.

    python scripts/run_v14.py train --seed 1 --out results/v14/n_seed1
    python scripts/run_v14.py train --seed 1 --level --out results/v14/n_level_seed1
    python scripts/run_v14.py train --seed 1 --limit 2 --max-epochs 1 --out <dir>      # dry run
    python scripts/run_v14.py posteriors --run results/v14/n_seed1     # val posteriors of a saved run again
    python scripts/run_v14.py baselines                                # V8 seed 0 and 1 on the same val features
    python scripts/run_v14.py score                                    # -> results/v14/val_table.md, val_results.json

Training is V8's recipe (configs/v14.yaml = configs/v8.yaml's settings) on vocabulary A plus N
(train_v14), on the 12-bin tuning features: RWC-P and GuitarSet from data/processed/v7_2_logcqt_Fs1, ChoCo
from its V8 cache. A run folder keeps model.pt, run.json, prior.json (61 classes, N last), n_counts.json
(chord -> N and N -> chord changes in the train labels), history.json and val_posteriors.npz.

Decoding: V8's frozen points (the network alone; fixed-lag Viterbi at lag 2 p_self 0.5, lag 4 p_self 0.8,
lag 8 p_self 0.98), prior power 0.25, the Billboard matrix with an N row and column
(transitions_v8.change_matrix with chord_to_n). Two decoder knobs for the N models: N's evidence times
n_weight (1, 2, 4; its tempered prior divided by it) and the counted chord -> N rate times n_rate_factor
(1, 0.5, 2). The network alone is the frame argmax, which no knob changes.

Baselines, on the same rebuilt val features: V8 A seed 0 and seed 1 (results/v8/A_seed<s>) and V11 seed 1
(results/v11/pool_kd1_seed1, posteriors from V13), alone and with decode_v10.level_gate at 0.005 (N where
the mean of the frame's log-CQT column is below it), at V8's points and at V11's own p_self (0.3, 0.7, 0.98).

Scores (validation only; RWC-P, GuitarSet, ChoCo and their mean, never pooled across sets), from the
frame_score tables of results/v10/cache/songs.npz (60 chords and N in column 60):
    with N        mir_eval majmin / sevenths as they are: reference N is scored, only an N output matches it
    N-excluded    reference N seconds left out
    N seconds     found (reference N, output N), missed (reference N, output a chord), false (output N on
                  a reference chord majmin scores)
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from chordtracker import (data_choco, data_v5, data_v7_2, data_v8, decode_v10, features_v14, frame_score, model_v14,
                          online_hmm, paths, train_v7_2, train_v14, transitions, transitions_v8)
from chordtracker.train_v14 import LevelSong

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config, evaluation_sets)
import run_v8  # noqa: E402  (model_for)
import run_v10  # noqa: E402  (load_songs: the frame_score tables)

CONFIG = paths.REPO_ROOT / "configs" / "v14.yaml"
OUT = paths.REPO_ROOT / "results" / "v14"
SETS = ("rwc", "guitarset", "choco")
SET_NAMES = {"rwc": "RWC-P", "guitarset": "GuitarSet", "choco": "ChoCo", "mean": "mean"}
METRICS = ("majmin", "sevenths")
N_RUNS = {"N seed 1": OUT / "n_seed1", "N seed 0": OUT / "n_seed0", "N + level seed 1": OUT / "n_level_seed1"}
MATCHING_V8 = {"N seed 1": "V8 seed 1", "N seed 0": "V8 seed 0", "N + level seed 1": "V8 seed 1"}
V8_RUNS = {s: paths.REPO_ROOT / "results" / "v8" / f"A_seed{s}" for s in (0, 1)}
V11_RUN = paths.REPO_ROOT / "results" / "v11" / "pool_kd1_seed1"
V11_POINTS = {"lag 2": 0.3, "lag 4": 0.7, "lag 8": 0.98}       # configs/v11.yaml frozen.points
GATE = 0.005
LAG8 = "lag 8"
POSTERIORS = "val_posteriors.npz"


def config() -> dict:
    return run_v5.load_config(CONFIG)


def n_chords(cfg: dict) -> int:
    return data_v8.n_chord_classes(cfg["vocab"])


def n_classes(cfg: dict) -> int:
    return n_chords(cfg) + (1 if cfg["with_n"] else 0)


def load_song(take, cfg: dict, level_input: bool) -> LevelSong:
    """The cached log-CQT (ChoCo's V8 cache, else the rebuilt 12-bin cache), A + N targets, and the level."""
    song = data_choco.load(take, cfg) if take.key.startswith("choco/") else data_v7_2.load(take, cfg)
    song = data_v8.retarget(song, take.annotation, cfg, cfg["vocab"], cfg["with_n"])
    level = features_v14.cached_levels(take.audio, take.key, cfg, song.cqt.shape[1]) if level_input else None
    return LevelSong(song.key, song.cqt, song.targets, level)


def train_takes(cfg: dict, first) -> list:
    return (first(data_v5.rwc_takes("train")) + first(data_v5.guitarset_takes("train"))
            + first(data_choco.choco_takes("train", cfg)))


def val_takes(cfg: dict, first=lambda xs: xs) -> dict[str, list]:
    return {**{n: first(t) for n, t in run_v5.evaluation_sets("val").items()},
            "choco": first(data_choco.choco_takes("val", cfg))}


# ---------------------------------------------------------------- train and posteriors

def train(args) -> None:
    cfg = config()
    cfg["train"].update(seed=args.seed, level_input=args.level)
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    train_songs = [load_song(t, cfg, args.level) for t in train_takes(cfg, first)]
    val_songs = {n: [load_song(t, cfg, args.level) for t in ts] for n, ts in val_takes(cfg, first).items()}
    targets = [s.targets for s in train_songs]
    print(f"train: {len(train_songs)} songs, vocab {cfg['vocab']} + N ({n_classes(cfg)} classes), seed {args.seed}, "
          f"level input {args.level}", flush=True)
    counts = transitions_v8.n_change_counts(targets, n_chords(cfg))
    prior = transitions_v8.training_prior(np.concatenate(targets), n_classes(cfg), n_chords(cfg))
    (out / "n_counts.json").write_text(json.dumps({**counts, "chord_to_n_rate": transitions_v8.n_rate(counts)}))
    (out / "prior.json").write_text(json.dumps(prior.tolist()))
    (out / "run.json").write_text(json.dumps({"n_classes": n_classes(cfg), "level_input": args.level,
                                              "train": cfg["train"]}, indent=2))
    print(f"N: {100 * prior[-1]:.2f} % of the train frames; changes {counts}", flush=True)
    model, history = train_v14.train(train_songs, val_songs, cfg, n_classes(cfg), n_chords(cfg))
    train_v14.save(model, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    best = max(history, key=lambda h: h["val_mean"])
    print(f"best epoch {best['epoch']} of {len(history)}, val frame accuracy (N included) "
          + "  ".join(f"{k[4:]} {100 * v:.2f}%" for k, v in best.items() if k.startswith("val_"))
          + f", {sum(h['seconds'] for h in history) / 60:.0f} min", flush=True)
    if not args.limit:
        save_posteriors(out, cfg)


def save_posteriors(run: Path, cfg: dict) -> None:
    """<run>/val_posteriors.npz: the (61, T) posterior of every val song, on the features it trained on."""
    model = model_v14.load(run, cfg).cpu().eval()
    out = {}
    for takes in val_takes(cfg).values():
        for take in takes:
            song = load_song(take, cfg, model.level_input)
            out[take.key] = train_v14.song_posteriors(model, song, cfg["train"]["input_lookahead"]).astype(np.float32)
    np.savez(run / POSTERIORS, **out)
    print(f"{run}: {len(out)} val posteriors", flush=True)


def baselines(_args) -> None:
    """results/v14/baselines/v8_seed<s>.npz: V8's (60, T) val posteriors on the rebuilt features."""
    cfg = config()
    (OUT / "baselines").mkdir(parents=True, exist_ok=True)
    for seed, run in V8_RUNS.items():
        model = run_v8.model_for(run, data_v8.n_chord_classes("A"), cfg).cpu().eval()
        out = {}
        for takes in val_takes(cfg).values():
            for take in takes:
                song = load_song(take, cfg, False)
                out[take.key] = train_v7_2.song_posteriors(model, song, cfg["train"]["input_lookahead"]).astype(np.float32)
        np.savez(OUT / "baselines" / f"v8_seed{seed}.npz", **out)
        print(f"V8 seed {seed}: {len(out)} val posteriors", flush=True)


# ---------------------------------------------------------------- decoding

def billboard_counts(cfg: dict) -> np.ndarray:
    return np.array(json.loads((paths.REPO_ROOT / cfg["transitions"]["counts"]).read_text())["counts"], dtype=float)


def specs(cfg: dict, prior: np.ndarray, p_selfs: dict[str, float], chord_to_n: float | None = None,
          n_weight: float = 1.0) -> list[tuple]:
    """[(point, A, decode prior, lag)] for the fixed-lag points; with chord_to_n the matrix has N's row and column."""
    t = cfg["transitions"]
    Q = transitions_v8.change_matrix(billboard_counts(cfg), cfg["vocab"], prior, t["alpha"], t["tau"], chord_to_n)
    decode_prior = transitions_v8.tempered_prior(prior, cfg["decoder"]["prior_power"])
    if chord_to_n is not None:
        decode_prior[-1] /= n_weight
        decode_prior = decode_prior / decode_prior.sum()
    return [(point, transitions.transition(Q, p), decode_prior, int(point.split()[1])) for point, p in p_selfs.items()]


def decode(job) -> dict[str, np.ndarray]:
    """{point: (T,) path}: 'network' is the frame argmax, then every fixed-lag spec."""
    post, point_specs = job
    out = {"network": post.argmax(axis=0)}
    for point, A, prior, lag in point_specs:
        out[point] = online_hmm.fixed_lag_viterbi(post, A, prior, [lag])[lag]
    return out


def decode_many(jobs: dict) -> dict:
    """{name: decode(job)} in parallel (one process per core); one after another where no pool is allowed."""
    keys = list(jobs)
    try:
        with ProcessPoolExecutor(8) as pool:
            results = list(pool.map(decode, [jobs[k] for k in keys], chunksize=2))
    except PermissionError:
        results = [decode(jobs[k]) for k in keys]
    return dict(zip(keys, results))


# ---------------------------------------------------------------- scoring

def score(paths_by_song: dict[str, dict], songs: dict[str, dict], n_label: int) -> dict:
    """{point: {set: {metric[_non]: %, ref_n, found_n, missed_n, false_n (s)}}}, plus set 'mean' for the metrics."""
    acc: dict = {}
    for key, by_point in paths_by_song.items():
        s = songs[key]
        for point, path in by_point.items():
            a = acc.setdefault(point, {}).setdefault(s["set"], {"ref_n": 0.0, "found_n": 0.0, "missed_n": 0.0,
                                                                "false_n": 0.0, **{f"{m}{v}": [0.0, 0.0] for m in METRICS
                                                                                   for v in ("", "_non")}})
            out_n = path == n_label
            ref_n = s["majmin_M"][:, n_label]                     # reference N seconds of each frame's cell
            a["ref_n"] += float(ref_n.sum())
            a["found_n"] += float(ref_n[out_n].sum())
            a["missed_n"] += float(ref_n[~out_n].sum())
            a["false_n"] += float((s["majmin_S"][:, n_label] - ref_n)[out_n].sum())
            for m in METRICS:
                for v in ("", "_non"):
                    matched, scored = frame_score.score_path(s[f"{m}{v}_M"], s[f"{m}{v}_S"], path)
                    a[f"{m}{v}"][0] += matched
                    a[f"{m}{v}"][1] += scored
    for by_set in acc.values():
        for a in by_set.values():
            for k, v in list(a.items()):
                if isinstance(v, list):
                    a[k] = 100 * v[0] / v[1]
        by_set["mean"] = {k: float(np.mean([by_set[d][k] for d in SETS])) for k in by_set[SETS[0]]
                          if k not in ("ref_n", "found_n", "missed_n", "false_n")}
    return acc


def load_npz(file: Path, prefix: str = "") -> dict[str, np.ndarray]:
    z = np.load(file)
    return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}


def score_all(_args) -> None:
    cfg = config()
    songs = run_v10.load_songs()
    n_label = n_chords(cfg)
    v8_points = {p["name"].split(" (")[0]: p["p_self"] for p in cfg["decoder"]["points"] if p["lag"] is not None}
    levels = {}                                                   # V10's gate level on the rebuilt features
    for takes in val_takes(cfg).values():
        for take in takes:
            levels[take.key] = load_song(take, cfg, False).cqt.mean(axis=0)

    jobs, meta = {}, {}
    for seed, run in V8_RUNS.items():
        post = load_npz(OUT / "baselines" / f"v8_seed{seed}.npz")
        sp = specs(cfg, run_v10.prior_of(run), v8_points)
        meta[f"V8 seed {seed}"] = {}
        jobs.update({(f"V8 seed {seed}", k): (p, sp) for k, p in post.items()})
    v11 = load_npz(V11_RUN / "val_posteriors_v13.npz", "chord|")
    for name, points in (("V11 seed 1", v8_points), ("V11 seed 1, own p_self", V11_POINTS)):
        sp = specs(cfg, run_v10.prior_of(V11_RUN), points)
        meta[name] = {}
        jobs.update({(name, k): (p, sp) for k, p in v11.items()})
    knobs = [(w, f) for w in cfg["decoder"]["n_weight"] for f in cfg["decoder"]["n_rate_factor"]]
    for name, run in N_RUNS.items():
        if not (run / POSTERIORS).exists():
            print(f"{name}: no {run / POSTERIORS}, skipped", flush=True)
            continue
        post = load_npz(run / POSTERIORS)
        prior = run_v10.prior_of(run)
        rate = json.loads((run / "n_counts.json").read_text())["chord_to_n_rate"]
        for w, f in knobs:
            label = f"{name} | N x{w:g}, rate x{f:g}"
            meta[label] = {"run": name, "n_weight": w, "rate_factor": f, "chord_to_n": rate * f}
            sp = specs(cfg, prior, v8_points, rate * f, w)
            jobs.update({(label, k): (p, sp) for k, p in post.items()})
    for (name, key), (p, _) in jobs.items():
        if p.shape[1] != songs[key]["majmin_M"].shape[0]:
            raise ValueError(f"{name} {key}: {p.shape[1]} posterior frames, {songs[key]['majmin_M'].shape[0]} in the tables")
    print(f"decoding {len(jobs)} (system, song) pairs", flush=True)
    decoded = decode_many(jobs)

    results = {}
    for name in meta:
        paths_ = {key: by for (n, key), by in decoded.items() if n == name}
        results[name] = {"meta": meta[name], "scores": score(paths_, songs, n_label)}
        if name.startswith("V11"):
            gated = {key: {pt: decode_v10.level_gate(path, levels[key], GATE, n_label) for pt, path in by.items()}
                     for key, by in paths_.items()}
            results[f"{name} + gate {GATE:g}"] = {"meta": {"gate": GATE}, "scores": score(gated, songs, n_label)}
    write(results, cfg, knobs)


# ---------------------------------------------------------------- tables

POINTS = ("network", "lag 2", "lag 4", "lag 8")


def go_no_go(results: dict, name: str, label: str) -> dict:
    """Phase 2 checks at lag 8: sevenths with N beats V11 seed 1 + gate (the better of its two p_self sets);
    N-excluded sevenths within 0.3 of the matching V8 seed; GuitarSet false N under 5 s."""
    r = results[label]["scores"][LAG8]
    gate = max(results[f"{g} + gate {GATE:g}"]["scores"][LAG8]["mean"]["sevenths"]
               for g in ("V11 seed 1", "V11 seed 1, own p_self"))
    v8 = results[MATCHING_V8[name]]["scores"][LAG8]["mean"]["sevenths_non"]
    checks = {"sevenths with N vs V11 + gate": (r["mean"]["sevenths"] - gate, r["mean"]["sevenths"] > gate),
              f"N-excluded sevenths vs {MATCHING_V8[name]}": (r["mean"]["sevenths_non"] - v8,
                                                             r["mean"]["sevenths_non"] >= v8 - 0.3),
              "GuitarSet false N s": (r["guitarset"]["false_n"], r["guitarset"]["false_n"] < 5.0)}
    return {"checks": checks, "go": all(ok for _, ok in checks.values())}


def write(results: dict, cfg: dict, knobs: list) -> None:
    default = {n: f"{n} | N x1, rate x1" for n in N_RUNS if f"{n} | N x1, rate x1" in results}
    main = [n for n in results if "|" not in n] + list(default.values())
    lines = ["# V14 phase 1: an N class, validation", "",
             "Validation only (RWC-P val, GuitarSet val, ChoCo val; mean = the mean of the three sets), the 12-bin "
             "tuning features for every system. Points: V8's frozen points (network alone; fixed-lag Viterbi lag 2 "
             "p_self 0.5, lag 4 p_self 0.8, lag 8 p_self 0.98), prior power 0.25, Billboard matrix; \"own p_self\" is "
             "V11's frozen 0.3 / 0.7 / 0.98. N models at the counted chord -> N rate and N weight 1 here; the knobs "
             "are in the last tables. Gate: decode_v10.level_gate at 0.005 (N where the mean log-CQT of the frame "
             "is below it).", ""]
    for variant, title in (("", "with N scored (mir_eval as it is)"), ("_non", "reference N excluded")):
        lines += [f"## majmin / sevenths %, {title}", "",
                  "| system | point | RWC-P | GuitarSet | ChoCo | mean |", "|---|---|---:|---:|---:|---:|"]
        for name in main:
            for point in POINTS:
                s = results[name]["scores"][point]
                cells = [f"{s[d][f'majmin{variant}']:.2f} / {s[d][f'sevenths{variant}']:.2f}" for d in (*SETS, "mean")]
                lines.append(f"| {name.split(' | ')[0]} | {point} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## N seconds", "", "Found: reference N output as N. Missed: reference N output as a chord. False: N "
              "output on a reference chord (one majmin scores). GuitarSet has no reference N.", "",
              "| system | point | RWC-P ref / found / missed / false | GuitarSet false | ChoCo ref / found / missed / false |",
              "|---|---|---:|---:|---:|"]
    for name in main:
        for point in POINTS:
            s = results[name]["scores"][point]
            nf = lambda d: f"{s[d]['ref_n']:.0f} / {s[d]['found_n']:.0f} / {s[d]['missed_n']:.0f} / {s[d]['false_n']:.0f}"
            lines.append(f"| {name.split(' | ')[0]} | {point} | {nf('rwc')} | {s['guitarset']['false_n']:.0f} | "
                         f"{nf('choco')} |")
    lines += ["", "## Decoder knobs (N models, lag 8)", "",
              "N weight: N's evidence times this. Rate: the counted chord -> N rate times this.", "",
              "| system | N weight | rate | chord -> N | mean sevenths with N | mean sevenths N-excl. | "
              "RWC-P found / false | ChoCo found / false | GuitarSet false |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in N_RUNS:
        for w, f in knobs:
            label = f"{name} | N x{w:g}, rate x{f:g}"
            if label not in results:
                continue
            s, m = results[label]["scores"][LAG8], results[label]["meta"]
            lines.append(f"| {name} | {w:g} | {f:g} | {m['chord_to_n']:.4f} | {s['mean']['sevenths']:.2f} | "
                         f"{s['mean']['sevenths_non']:.2f} | {s['rwc']['found_n']:.0f} / {s['rwc']['false_n']:.0f} | "
                         f"{s['choco']['found_n']:.0f} / {s['choco']['false_n']:.0f} | {s['guitarset']['false_n']:.0f} |")
    lines += ["", "## Go / no-go for phase 2 (lag 8)", "",
              "Checks: 3-set mean sevenths with N above V11 seed 1 + gate (the better of its two p_self sets); "
              "3-set mean N-excluded sevenths at most 0.3 below the matching V8 seed; GuitarSet false N under 5 s. "
              "At the default knobs (N weight 1, counted rate), and at the knob setting with the best mean sevenths "
              "with N (chosen on validation, so optimistic).", "",
              "| system | knobs | vs V11 + gate | N-excl. vs V8 | GuitarSet false N s | go |", "|---|---|---:|---:|---:|---|"]
    verdicts = {}
    for name in N_RUNS:
        labels = [f"{name} | N x{w:g}, rate x{f:g}" for w, f in knobs]
        labels = [lb for lb in labels if lb in results]
        if not labels:
            continue
        best = max(labels, key=lambda lb: results[lb]["scores"][LAG8]["mean"]["sevenths"])
        for which, label in (("default", labels[0]), ("best", best)):
            v = go_no_go(results, name, label)
            verdicts[f"{name} ({which})"] = {"label": label, **v}
            c = list(v["checks"].values())
            mark = lambda x: "pass" if x else "fail"
            lines.append(f"| {name} | {label.split(' | ')[1]} | {c[0][0]:+.2f} ({mark(c[0][1])}) | {c[1][0]:+.2f} "
                         f"({mark(c[1][1])}) | {c[2][0]:.1f} ({mark(c[2][1])}) | {'GO' if v['go'] else 'NO-GO'} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "val_table.md").write_text("\n".join(lines) + "\n")
    (OUT / "val_results.json").write_text(json.dumps({"results": results, "go_no_go": verdicts}, indent=1,
                                                     default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["train", "posteriors", "baselines", "score"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", action="store_true", help="the level variant: the GRU also reads the input level")
    parser.add_argument("--out", type=Path, help="run folder (train)")
    parser.add_argument("--run", type=Path, help="run folder (posteriors)")
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int, help="dry run: override train.max_epochs")
    args = parser.parse_args(argv)
    if args.step == "train":
        if args.out is None:
            parser.error("train needs --out")
        train(args)
    elif args.step == "posteriors":
        save_posteriors(args.run, config())
    elif args.step == "baselines":
        baselines(args)
    else:
        score_all(args)


if __name__ == "__main__":
    main()
