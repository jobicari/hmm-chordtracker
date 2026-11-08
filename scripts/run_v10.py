"""V10 on validation: decoder-side changes on V8's saved models (vocabulary A), scored on the three val sets.

    python scripts/run_v10.py cache --run results/v8/A_seed0      # posteriors of a run -> <run>/val_posteriors.npz
    python scripts/run_v10.py ensemble                            # step 1: seed 0 + seed 1 posteriors
    python scripts/run_v10.py offsets                             # step 2: per-quality log offsets
    python scripts/run_v10.py retune                              # step 3: val songs whose tuning estimate moved
    python scripts/run_v10.py gate                                # step 4: N where the input level is low
    python scripts/run_v10.py points --run results/v10/<run>      # the V8 points of any saved A run

Everything is validation (RWC-P val, GuitarSet val, ChoCo val, never pooled). The V8 points: network alone
(frame argmax, 0.28 s) and fixed-lag Viterbi at lag 2, 4 and 8 (0.46, 0.65, 1.02 s) over
configs/v8.yaml's p_self grid with the Billboard matrix; per point the p_self with the best mean sevenths
of the three sets is reported, as V8 did. Scores are aligned WCSR (mir_eval majmin and sevenths) through
frame_score's tables, which equal evaluate.score; `cache` stores the tables once in results/v10/cache/.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from chordtracker import (data_choco, data_v5, decode_v10, features_v7_1, frame_score, model_v10, paths, train_v7_2,
                          transitions, transitions_v8)

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config, evaluation_sets)
import run_v8  # noqa: E402  (load_song, decoder_parts, class_labels, model_for)

VOCAB = "A"
SETS = ("rwc", "guitarset", "choco")
METRICS = ("majmin", "sevenths")
LAGS = [2, 4, 8]
QUALITIES = list(run_v8.qualities(VOCAB))           # maj, min, 7, maj7, min7
N_CLASSES = run_v8.n_classes(VOCAB)
CACHE = paths.REPO_ROOT / "results" / "v10" / "cache"
SEEDS = {s: paths.REPO_ROOT / "results" / "v8" / f"A_seed{s}" for s in (0, 1)}


def config() -> dict:
    return run_v5.load_config(paths.REPO_ROOT / "configs" / "v8.yaml")


def val_takes(cfg: dict) -> dict[str, list]:
    return {**run_v5.evaluation_sets("val"), "choco": data_choco.choco_takes("val", cfg)}


# ---------------------------------------------------------------- cache

def no_n(annotation):
    """The annotation with reference N as X, so the metrics leave it out (the N-excluded variant)."""
    return type(annotation)(annotation.intervals, ["X" if label == "N" else label for label in annotation.labels])


def build_song_cache(cfg: dict) -> None:
    """results/v10/cache/songs.npz: per val song its set, targets, frame level and the frame_score tables.

    Tables for the 60 classes and N (column 60), for majmin and sevenths, as V8 scores them (reference N is
    scored) and N-excluded (reference N left out). Level: the mean of the frame's log-CQT column.
    """
    out, labels = {}, run_v8.class_labels(VOCAB) + ["N"]
    period = cfg["cqt"]["hop_length"] / cfg["audio"]["sr"]
    for dataset, takes in val_takes(cfg).items():
        for take in takes:
            song = run_v8.load_song(take, cfg, VOCAB)
            n = song.cqt.shape[1]
            out[f"{take.key}|set"] = np.array(dataset)
            out[f"{take.key}|targets"] = song.targets
            out[f"{take.key}|level"] = song.cqt.mean(axis=0)
            ref_n = sum(e - s for (s, e), lab in zip(take.annotation.intervals, take.annotation.labels) if lab == "N")
            out[f"{take.key}|ref_n_s"] = np.array(ref_n)
            for metric in METRICS:
                for variant, ann in (("", take.annotation), ("_non", no_n(take.annotation))):
                    M, S = frame_score.tables(ann, labels, n, period, metric)
                    out[f"{take.key}|{metric}{variant}_M"] = M.astype(np.float32)
                    out[f"{take.key}|{metric}{variant}_S"] = S.astype(np.float32)
        print(f"{dataset}: {len(takes)} takes cached", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "songs.npz", **out)


def load_songs() -> dict[str, dict]:
    """{song key: {set, targets, level, ref_n_s, <metric>[_non]_M / _S}} from results/v10/cache/songs.npz."""
    z = np.load(CACHE / "songs.npz")
    songs: dict[str, dict] = {}
    for name in z.files:
        key, field = name.split("|")
        songs.setdefault(key, {})[field] = z[name] if z[name].ndim else z[name].item()
    return songs


def cache_posteriors(run: Path, cfg: dict) -> None:
    """<run>/val_posteriors.npz: the (60, T) network posterior of every val song (eval mode, unshifted)."""
    model = model_v10.load(run, N_CLASSES, cfg)
    out = {}
    for takes in val_takes(cfg).values():
        for take in takes:
            song = run_v8.load_song(take, cfg, VOCAB)
            out[take.key] = train_v7_2.song_posteriors(model, song, cfg["train"]["input_lookahead"]).astype(np.float32)
    np.savez(run / "val_posteriors.npz", **out)
    print(f"{run}: {len(out)} posteriors", flush=True)


def load_posteriors(run: Path) -> dict[str, np.ndarray]:
    z = np.load(run / "val_posteriors.npz")
    return {k: z[k] for k in z.files}


def prior_of(run: Path) -> np.ndarray:
    return np.array(json.loads((run / "prior.json").read_text()))


# ---------------------------------------------------------------- decoding and scoring

_WORKER: dict = {}


def _init(A_of, prior, lags):
    _WORKER.update(A_of=A_of, prior=prior, lags=lags)


def _decode(post):
    return decode_v10.decode_points(post, _WORKER["A_of"], _WORKER["prior"], _WORKER["lags"])


def transition_mats(cfg: dict, prior: np.ndarray, p_selfs: list[float]) -> dict[float, np.ndarray]:
    Q = run_v8.decoder_parts(cfg, VOCAB, prior)
    return {p: transitions.transition(Q, p) for p in p_selfs}


def decode_all(posts: dict[str, np.ndarray], cfg: dict, prior: np.ndarray, power: float,
               p_selfs: list[float] | None = None, lags: list[int] = LAGS) -> dict[str, dict]:
    """{song: {(point, p_self): path}} for every song, in parallel. prior: the training prior (Q and division)."""
    p_selfs = cfg["decoder"]["p_self_grid"] if p_selfs is None else p_selfs
    A_of = transition_mats(cfg, prior, p_selfs)
    decode_prior = transitions_v8.tempered_prior(prior, power)
    keys = list(posts)
    with ProcessPoolExecutor(8, initializer=_init, initargs=(A_of, decode_prior, lags)) as pool:
        results = list(pool.map(_decode, [posts[k] for k in keys], chunksize=4))
    return dict(zip(keys, results))


def pooled(paths_by_song: dict[str, dict], songs: dict[str, dict], variant: str = "",
           sets=SETS) -> dict[tuple, dict[str, float]]:
    """{(set, point, p_self): {metric: WCSR}} pooled per set; set 'mean' is the mean over the sets."""
    acc: dict[tuple, dict[str, list[float]]] = {}
    for key, by_point in paths_by_song.items():
        s = songs[key]
        if s["set"] not in sets:
            continue
        for point, path in by_point.items():
            a = acc.setdefault((s["set"], *point), {m: [0.0, 0.0] for m in METRICS})
            for m in METRICS:
                matched, scored = frame_score.score_path(s[f"{m}{variant}_M"], s[f"{m}{variant}_S"], path)
                a[m][0] += matched
                a[m][1] += scored
    out = {k: {m: v[m][0] / v[m][1] for m in METRICS} for k, v in acc.items()}
    for point in {k[1:] for k in out}:
        present = [out[(d, *point)] for d in sets if (d, *point) in out]
        out[("mean", *point)] = {m: float(np.mean([r[m] for r in present])) for m in METRICS}
    return out


def best_points(scores: dict[tuple, dict], select_on: str = "mean") -> dict[str, float | None]:
    """{point: p_self} with the best `select_on` sevenths per point (V8's rule on the 3-set mean)."""
    best: dict[str, tuple[float, float | None]] = {}
    for (dataset, point, p_self), r in scores.items():
        if dataset == select_on and (point not in best or r["sevenths"] > best[point][0] + 1e-12):
            best[point] = (r["sevenths"], p_self)
    return {point: p for point, (_, p) in best.items()}


POINT_ORDER = ["network", "lag 2", "lag 4", "lag 8"]
POINT_NAMES = {"network": "network alone (0.28 s)", "lag 2": "lag 2 (0.46 s)", "lag 4": "lag 4 (0.65 s)",
               "lag 8": "lag 8 (1.02 s)"}


def point_table(scores: dict[tuple, dict], chosen: dict[str, float | None] | None = None) -> list[dict]:
    """One row per V8 point at its chosen p_self: every set and the mean, majmin and sevenths in percent."""
    chosen = best_points(scores) if chosen is None else chosen
    rows = []
    for point in POINT_ORDER:
        if point not in chosen:
            continue
        p = chosen[point]
        row = {"point": point, "p_self": p}
        for d in (*SETS, "mean"):
            r = scores.get((d, point, p))
            if r:
                row[d] = (100 * r["majmin"], 100 * r["sevenths"])
        rows.append(row)
    return rows


def markdown(title: str, tables: dict[str, list[dict]]) -> str:
    """A markdown table: one row per (system, point), cells majmin / sevenths %."""
    lines = [f"| {title} | point | p_self | RWC-P | GuitarSet | ChoCo | mean |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, rows in tables.items():
        for r in rows:
            cells = [f"{r[d][0]:.2f} / {r[d][1]:.2f}" if d in r else "" for d in (*SETS, "mean")]
            p = "" if r["p_self"] is None else f"{r['p_self']:g}"
            lines.append(f"| {name} | {POINT_NAMES[r['point']]} | {p} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write(name: str, text: str, data) -> None:
    out = paths.REPO_ROOT / "results" / "v10" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text(text + "\n")
    (out / "results.json").write_text(json.dumps(data, indent=1, default=str))
    print(text, flush=True)


def serial(scores: dict[tuple, dict]) -> list[dict]:
    return [{"set": k[0], "point": k[1], "p_self": k[2], **v} for k, v in scores.items()]


# ---------------------------------------------------------------- step 1

def step_ensemble(cfg: dict, songs: dict) -> None:
    posts = {s: load_posteriors(SEEDS[s]) for s in SEEDS}
    priors = {s: prior_of(SEEDS[s]) for s in SEEDS}
    prior = (priors[0] + priors[1]) / 2                 # training prior: same targets, so identical in practice
    tables, data = {}, {}
    for power in (0.25, 0.0, 0.5):
        singles = {}
        for s in SEEDS:
            sc = pooled(decode_all(posts[s], cfg, priors[s], power), songs)
            singles[s] = sc
            tables[f"seed {s}, k {power:g}"] = point_table(sc)
            data[f"seed{s}_pp{power:g}"] = serial(sc)
        tables[f"two-seed average, k {power:g}"] = average_rows(point_table(singles[0]), point_table(singles[1]))
        for mode in ("geometric", "arithmetic"):
            combined = {k: decode_v10.ensemble([posts[0][k], posts[1][k]], mode) for k in posts[0]}
            sc = pooled(decode_all(combined, cfg, prior, power), songs)
            tables[f"{mode} ensemble, k {power:g}"] = point_table(sc)
            data[f"{mode}_pp{power:g}"] = serial(sc)
        print(f"prior power {power:g} done", flush=True)
    write("step1_ensemble", markdown("system", tables), data)


def average_rows(a: list[dict], b: list[dict]) -> list[dict]:
    """Per point the mean of two seeds' rows (each at its own chosen p_self); p_self left blank."""
    out = []
    for ra, rb in zip(a, b):
        row = {"point": ra["point"], "p_self": None}
        for d in (*SETS, "mean"):
            row[d] = tuple((x + y) / 2 for x, y in zip(ra[d], rb[d]))
        out.append(row)
    return out


# ---------------------------------------------------------------- step 3

def step_retune(cfg: dict, songs: dict) -> None:
    """Val songs whose tuning estimate moved by more than 5 cents (results/v10/step3_tuning): their causal
    log-CQT again with the fixed estimate (results/v10/cache/retuned/, the shared cache is left as it is),
    then seed 0 and seed 1 scored on val with those songs replaced. No retraining."""
    moved = [row["key"] for row in json.loads((paths.REPO_ROOT / "results" / "v10" / "step3_tuning" / "estimates.json")
                                              .read_text())
             if row["split"] == "val" and abs(row["new_cents"] - row["old_cents"]) > 5]
    takes = {t.key: t for ts in val_takes(cfg).values() for t in ts}
    out = CACHE / "retuned"
    out.mkdir(parents=True, exist_ok=True)
    new_songs = {}
    for key in moved:
        file = out / f"{key.replace('/', '__')}.npy"
        if not file.exists():
            np.save(file, features_v7_1.causal_log_cqt(takes[key].audio, cfg).astype(np.float16))
        cqt = np.load(file).astype(np.float32)
        old = run_v8.load_song(takes[key], cfg, VOCAB)
        assert cqt.shape == old.cqt.shape, key
        new_songs[key] = data_v5.Song(key, cqt, old.targets)
        print(f"{key}: mean |change| of the log-CQT {np.abs(cqt - old.cqt).mean():.4f}", flush=True)
    tables, data = {}, {}
    for s in SEEDS:
        posts = load_posteriors(SEEDS[s])
        model = model_v10.load(SEEDS[s], N_CLASSES, cfg)
        retuned = {**posts, **{k: train_v7_2.song_posteriors(model, song, cfg["train"]["input_lookahead"])
                               for k, song in new_songs.items()}}
        for name, p in (("cached features", posts), ("retuned features", retuned)):
            sc = pooled(decode_all(p, cfg, prior_of(SEEDS[s]), cfg["decoder"]["prior_power"]), songs)
            tables[f"seed {s}, {name}"] = point_table(sc)
            data[f"seed{s}_{name}"] = serial(sc)
    write("step3_retune", f"Val songs re-cached: {', '.join(moved)}\n\n" + markdown("features", tables), data)


# ---------------------------------------------------------------- main

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["cache", "ensemble", "offsets", "retune", "gate", "points"])
    parser.add_argument("--run", type=Path, help="a saved vocabulary-A run (model.pt, prior.json)")
    parser.add_argument("--name", help="results/v10/<name> for the points step; default the run's folder name")
    args = parser.parse_args(argv)
    cfg = config()
    if args.step == "cache":
        if not (CACHE / "songs.npz").exists():
            build_song_cache(cfg)
        if args.run:
            cache_posteriors(args.run, cfg)
        return
    songs = load_songs()
    if args.step == "ensemble":
        step_ensemble(cfg, songs)
    elif args.step == "retune":
        step_retune(cfg, songs)
    elif args.step == "offsets":
        import v10_offsets
        v10_offsets.run(cfg, songs)
    elif args.step == "gate":
        import v10_gate
        v10_gate.run(cfg, songs)
    elif args.step == "points":
        if not (args.run / "val_posteriors.npz").exists():
            cache_posteriors(args.run, cfg)
        sc = pooled(decode_all(load_posteriors(args.run), cfg, prior_of(args.run), cfg["decoder"]["prior_power"]),
                    songs)
        write(args.name or f"points_{args.run.name}", markdown("run", {args.run.name: point_table(sc)}), serial(sc))


if __name__ == "__main__":
    main()
