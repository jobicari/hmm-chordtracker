"""Run V15: V14's N model fine-tuned on vocabulary A+ (dim, aug, hdim7), scored on validation.

    python scripts/run_v15.py tables                                    # val scoring tables (once, cached)
    python scripts/run_v15.py train --seed 0 --out results/v15/ft_seed0
    python scripts/run_v15.py train --seed 0 --replay 4 --focus 4 --out results/v15/ft_r50_seed0
    python scripts/run_v15.py train --seed 0 --replay 0 --focus 8 --out results/v15/ft_focus_seed0   # no replay
    python scripts/run_v15.py train --seed 0 --limit 3 --max-epochs 1 --replay-excerpts 16 --out <dir>  # dry run
    python scripts/run_v15.py posteriors --run results/v15/ft_seed0     # val posteriors of a saved run again
    python scripts/run_v15.py score                                     # -> results/v15/val_table.md, val_results.json

Fine-tune (train_v15, configs/v15.yaml): every weight of results/v14/n_seed0 copied, 36 new output rows
(C:dim .. B:hdim7, after N; dim7 trained as dim), replay of the normal train data plus oversampled focus
excerpts around every new-quality chord (12 shifts in turn, tempo 1 +/- 0.06, gain +/- 3 dB), lr 1e-4
(output layer 5e-4), the epoch chosen on validation (train_v15.criterion). A run folder keeps model.pt
(best criterion), model_best_sevenths.pt, run.json, prior.json (natural train prior), mix_prior.json (the
fine-tune's trained target mix), n_counts.json, history.json and val_posteriors.npz /
val_posteriors_best_sevenths.npz, all priors and posteriors in decode order (data_v15: the chords, N last).

Validation tables (data/v15_cache/val_tables.npz): frame_score tables of majmin, sevenths, tetrads and
mirex over the 97 decode labels (reference N scored, as V14's "with N"), and the A+ frame targets, on the
same features V14 n_seed0 was scored on.

Scores (validation only; RWC-P, GuitarSet, ChoCo and their mean, never pooled across sets), the
network alone and lag 8 at V14 n_seed0's demo point (p_self 0.98, N weight 0.1, chord -> N rate x0.5,
prior power 0.25), for V14 n_seed0 (its 61 classes, decoded as run_v14 does) and every V15 run, whose
evidence divides by the trained target mix or by the natural train prior. Per-quality recall and
precision (eval_v15) pooled over the three sets.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from chordtracker import (data_choco, data_v7_2, data_v15, eval_v15, model_v14, paths, train_v6, train_v14,
                          train_v15, transitions_v8)

sys.path.insert(0, str(Path(__file__).parent))
import run_v14  # noqa: E402  (config, train_takes, val_takes, billboard_counts)

CONFIG = paths.REPO_ROOT / "configs" / "v15.yaml"
OUT = paths.REPO_ROOT / "results" / "v15"
TABLES = paths.DATA_DIR / "v15_cache" / "val_tables.npz"
SET_NAMES = {"rwc": "RWC-P", "guitarset": "GuitarSet", "choco": "ChoCo", "mean": "mean"}
RUNS = {"V15 seed 0 (62.5 % replay)": OUT / "ft_seed0", "V15 seed 0 (50 % replay)": OUT / "ft_r50_seed0",
        "V15 seed 0 (focus only)": OUT / "ft_focus_seed0"}
OLD = ("maj", "min", "7", "maj7", "min7")


def config() -> dict:
    """V14's settings (configs/v14.yaml on v2.yaml) with V15's sections on top."""
    return {**run_v14.config(), **yaml.safe_load(CONFIG.read_text())}


def period(cfg: dict) -> float:
    return cfg["cqt"]["hop_length"] / cfg["audio"]["sr"]


def load_song(take, cfg: dict, vocab: data_v15.Vocab):
    """V14's features (ChoCo's V8 cache, else the 12-bin cache) with A+ targets in model order."""
    song = data_choco.load(take, cfg) if take.key.startswith("choco/") else data_v7_2.load(take, cfg)
    return vocab.retarget(song, take.annotation, cfg)


def parallel(fn, jobs: list) -> list:
    """[fn(job)] on 8 processes; one after another where no process pool is allowed."""
    try:
        with ProcessPoolExecutor(8) as pool:
            return list(pool.map(fn, jobs, chunksize=2))
    except PermissionError:
        return [fn(j) for j in jobs]


# ---------------------------------------------------------------- tables

def _tables_job(job):
    key, ann, n, vocab, p = job
    return key, eval_v15.song_tables(ann, n, vocab, p)


def build_tables(_args) -> None:
    cfg = config()
    vocab = data_v15.vocab_from_cfg(cfg)
    jobs, meta = [], {}
    for name, takes in run_v14.val_takes(cfg).items():
        for take in takes:
            song = load_song(take, cfg, vocab)
            meta[take.key] = (name, vocab.to_decode(song.targets))
            jobs.append((take.key, take.annotation, song.cqt.shape[1], vocab, period(cfg)))
    print(f"building tables for {len(jobs)} val songs, {vocab.n_classes} classes", flush=True)
    out = {}
    for key, tables in parallel(_tables_job, jobs):
        out[f"{key}|set"] = np.array(meta[key][0])
        out[f"{key}|targets"] = meta[key][1]
        out.update({f"{key}|{k}": v for k, v in tables.items()})
    TABLES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TABLES, labels=np.array(vocab.decode_names()), **out)
    print(f"{TABLES}: {len(jobs)} songs", flush=True)


def load_tables(metrics=eval_v15.METRICS) -> dict[str, dict]:
    """{song key: {set, targets, <metric>_M / _S}} for the metrics asked for."""
    z = np.load(TABLES)
    want = {"set", "targets"} | {f"{m}_{x}" for m in metrics for x in "MS"}
    songs: dict[str, dict] = {}
    for name in z.files:
        if "|" not in name:
            continue
        key, field = name.split("|")
        if field in want:
            songs.setdefault(key, {})[field] = z[name] if z[name].ndim else z[name].item()
    return songs


# ---------------------------------------------------------------- decoding

def mix_prior(counts: np.ndarray, vocab: data_v15.Vocab) -> np.ndarray:
    """(n_classes,) decode-order prior from model-order target counts, each quality's mass spread over its 12
    roots (the counts come from randomly shifted excerpts; this removes what is left of the key bias)."""
    c = counts[vocab.decode_order].astype(float)
    chords = np.repeat(c[:-1].reshape(-1, 12).mean(axis=1), 12)
    p = np.concatenate([chords, c[-1:]])
    return p / p.sum()


def demo_decoder(cfg: dict, qualities, prior: np.ndarray, divide: np.ndarray, rate: float):
    d = cfg["decoder"]
    t = cfg["transitions"]
    return eval_v15.decoder(qualities, run_v14.billboard_counts(cfg), prior, divide, d["p_self"],
                            rate * d["n_rate_factor"], d["n_weight"], d["prior_power"], t["alpha"], t["tau"])


def _decode_job(job):
    key, post, A, prior, lag = job
    return key, eval_v15.decode(post, A, prior, lag)


def decode_all(posts: dict[str, np.ndarray], A, prior, lag) -> dict[str, np.ndarray]:
    return dict(parallel(_decode_job, [(k, p, A, prior, lag) for k, p in posts.items()]))


def val_posteriors(model, songs: dict, vocab: data_v15.Vocab, lookahead: int) -> dict[str, np.ndarray]:
    """{key: (n_classes, T) decode-order posteriors} of the val songs, eval mode."""
    out = {}
    for key, song in songs.items():
        p = train_v14.song_posteriors(model, train_v14.LevelSong(song.key, song.cqt, song.targets), lookahead)
        out[key] = p[vocab.decode_order].astype(np.float32)
    return out


# ---------------------------------------------------------------- train

def train(args) -> None:
    cfg = config()
    t = cfg["finetune"]
    t["seed"] = args.seed
    if args.replay is not None:
        t["replay_per_batch"] = args.replay
    if args.focus is not None:
        t["focus_per_batch"] = args.focus
    if args.max_epochs:
        t["max_epochs"] = args.max_epochs
    if args.replay_excerpts:
        t["replay_excerpts"] = args.replay_excerpts
    if t["replay_per_batch"] == 0:                     # focus only: as many batches as the default 5 + 3 epoch
        t["batches_per_epoch"] = int(np.ceil(t["replay_excerpts"] / 5))
    vocab = data_v15.vocab_from_cfg(cfg)
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    train_songs = [load_song(tk, cfg, vocab) for tk in run_v14.train_takes(cfg, first)]
    if args.limit:                    # a dry run still needs focus runs: add the first train songs that have one
        extra = [tk for tk in run_v14.train_takes(cfg, lambda xs: xs[:40])
                 if (load_song(tk, cfg, vocab).targets > data_v15.N_BASE).any()][:2]
        train_songs += [load_song(tk, cfg, vocab) for tk in extra]
    focus = train_v15.focus_segments(train_songs, vocab)
    dec_targets = [vocab.to_decode(s.targets) for s in train_songs]
    prior = transitions_v8.training_prior(np.concatenate(dec_targets), vocab.n_classes, vocab.n_chords)
    counts = transitions_v8.n_change_counts(dec_targets, vocab.n_chords)
    rate = transitions_v8.n_rate(counts)
    print(f"train: {len(train_songs)} songs, {len(focus)} focus runs "
          + ", ".join(f"{q} {sum(1 for f in focus if f[3] == j)}" for j, q in enumerate(vocab.new))
          + f"; {vocab.n_classes} classes, seed {args.seed}, replay {t['replay_per_batch']} + focus "
          f"{t['focus_per_batch']} per batch", flush=True)
    (out / "prior.json").write_text(json.dumps(prior.tolist()))
    (out / "n_counts.json").write_text(json.dumps({**counts, "chord_to_n_rate": rate}))
    (out / "run.json").write_text(json.dumps({"n_classes": vocab.n_classes, "level_input": False,
                                              "vocab_plus": {"new": list(vocab.new), "merge": vocab.merge},
                                              "base_run": cfg["base_run"], "finetune": t}, indent=2))

    base = model_v14.load(paths.REPO_ROOT / cfg["base_run"], cfg)
    model_prior = prior[np.argsort(vocab.decode_order)]                    # decode -> model order
    model = train_v15.extend(base.cpu(), vocab, model_prior).to(next(base.parameters()).device)

    val_takes = {take.key: take for takes in run_v14.val_takes(cfg, first).values() for take in takes}
    val_songs = {k: load_song(tk, cfg, vocab) for k, tk in val_takes.items()}
    tables = load_tables(("sevenths",))
    tables = {k: v for k, v in tables.items() if k in val_songs}

    def evaluate(m, mix_counts):
        posts = val_posteriors(m, val_songs, vocab, t["input_lookahead"])
        divide = prior if mix_counts is None else mix_prior(mix_counts, vocab)
        A, d = demo_decoder(cfg, vocab.qualities, prior, divide, rate)
        net = eval_v15.score({k: p.argmax(axis=0) for k, p in posts.items()}, tables, vocab, period(cfg),
                             ("sevenths",))
        lag = eval_v15.score(decode_all(posts, A, d, cfg["decoder"]["lag"]), tables, vocab, period(cfg),
                             ("sevenths",))
        out_ = {"sevenths": lag["mean"]["sevenths"], "recall": eval_v15.new_recall(lag, vocab),
                "net_sevenths": net["mean"]["sevenths"], "net_recall": eval_v15.new_recall(net, vocab)}
        for q in vocab.new:
            out_[f"recall_{q}"] = lag["quality"][q]["recall"]
            out_[f"precision_{q}"] = lag["quality"][q]["precision"]
        return {k: float(np.nan_to_num(v)) for k, v in out_.items()}

    best, best_sev, history, mix = train_v15.finetune(model, train_songs, focus, t, vocab, evaluate)
    train_v6.save(best, out / "model.pt")
    train_v6.save(best_sev, out / "model_best_sevenths.pt")
    (out / "mix_prior.json").write_text(json.dumps(mix_prior(mix, vocab).tolist()))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    b = max(history, key=lambda h: h["criterion"])
    s = max(history, key=lambda h: h["sevenths"])
    print(f"best criterion epoch {b['epoch']} ({b['criterion']:.2f}: sevenths {b['sevenths']:.2f}, new recall "
          f"{b['recall']:.2f}); best sevenths epoch {s['epoch']} ({s['sevenths']:.2f}); "
          f"{sum(h['seconds'] for h in history) / 60:.0f} min", flush=True)
    if not args.limit:
        save_posteriors(out, cfg)


def load_run(run: Path, cfg: dict, weights: str = "model.pt"):
    info = json.loads((run / "run.json").read_text())
    model = model_v14.ChordCRNN(info["n_classes"], level_input=False, dropout=cfg["train"]["dropout"])
    model.load_state_dict(torch.load(run / weights, map_location="cpu"))
    return model.eval()


def save_posteriors(run: Path, cfg: dict) -> None:
    """<run>/val_posteriors.npz (model.pt) and val_posteriors_best_sevenths.npz, decode order."""
    vocab = data_v15.vocab_from_cfg(cfg)
    songs = {take.key: load_song(take, cfg, vocab) for takes in run_v14.val_takes(cfg).values() for take in takes}
    for weights, name in (("model.pt", "val_posteriors.npz"), ("model_best_sevenths.pt",
                                                              "val_posteriors_best_sevenths.npz")):
        posts = val_posteriors(load_run(run, cfg, weights), songs, vocab, cfg["finetune"]["input_lookahead"])
        np.savez(run / name, **posts)
        print(f"{run / name}: {len(posts)} val posteriors", flush=True)


# ---------------------------------------------------------------- score

def score_all(_args) -> None:
    cfg = config()
    vocab = data_v15.vocab_from_cfg(cfg)
    songs = load_tables()
    p = period(cfg)
    d = cfg["decoder"]
    systems = {}

    base = paths.REPO_ROOT / cfg["base_run"]                          # V14 n_seed0: 61 classes, as run_v14 decodes
    v14 = run_v14.load_npz(base / "val_posteriors.npz")
    v14_prior = np.array(json.loads((base / "prior.json").read_text()))
    v14_rate = json.loads((base / "n_counts.json").read_text())["chord_to_n_rate"]
    A, pr = demo_decoder(cfg, data_v15.BASE, v14_prior, v14_prior, v14_rate)
    systems["V14 n_seed0"] = {
        "network": {k: eval_v15.from_v14(x.argmax(axis=0), vocab) for k, x in v14.items()},
        "lag 8": {k: eval_v15.from_v14(x, vocab) for k, x in decode_all(v14, A, pr, d["lag"]).items()}}

    for name, run in RUNS.items():
        for weights, tag in (("val_posteriors.npz", ""), ("val_posteriors_best_sevenths.npz", ", best sevenths")):
            if not (run / weights).exists():
                print(f"{name}: no {run / weights}, skipped", flush=True)
                continue
            post = run_v14.load_npz(run / weights)
            prior = np.array(json.loads((run / "prior.json").read_text()))
            rate = json.loads((run / "n_counts.json").read_text())["chord_to_n_rate"]
            divides = {"mix": np.array(json.loads((run / "mix_prior.json").read_text())), "natural": prior}
            entry = {"network": {k: x.argmax(axis=0) for k, x in post.items()}}
            for src in d["prior_source"]:
                A, pr = demo_decoder(cfg, vocab.qualities, prior, divides[src], rate)
                entry[f"lag 8, {src} prior"] = decode_all(post, A, pr, d["lag"])
            systems[f"{name}{tag}"] = entry
    results = {name: {point: eval_v15.score(paths_, songs, vocab, p) for point, paths_ in by.items()}
               for name, by in systems.items()}
    write(results, vocab)


def write(results: dict, vocab: data_v15.Vocab) -> None:
    lines = ["# V15: dim, aug and hdim7 on top of V14's N model, validation", "",
             "Validation only (RWC-P val, GuitarSet val, ChoCo val; mean = the mean of the three sets), the features "
             "V14 n_seed0 is scored on. Lag 8 is V14 n_seed0's demo point: fixed-lag Viterbi lag 8, p_self 0.98, N "
             "weight 0.1, chord -> N rate x0.5, prior power 0.25; V15 runs divide by the fine-tune's target mix or by "
             "the natural train prior. Reference N is scored (V14's \"with N\"). majmin and sevenths leave dim, aug, "
             "dim7 and hdim7 references out, so the new classes can only cost them; tetrads scores every quality "
             "exactly, mirex counts an output sharing three notes with the reference as right.", ""]
    for metric in eval_v15.METRICS:
        lines += [f"## {metric} %", "", "| system | point | RWC-P | GuitarSet | ChoCo | mean |", "|---|---|---:|---:|---:|---:|"]
        for name, by in results.items():
            for point, r in by.items():
                cells = [f"{r['sets'][s][metric]:.2f}" for s in eval_v15.SETS] + [f"{r['mean'][metric]:.2f}"]
                lines.append(f"| {name} | {point} | " + " | ".join(cells) + " |")
        lines.append("")
    qs = OLD + vocab.new
    ref_s = next(iter(next(iter(results.values())).values()))["quality"]
    lines += ["## Per quality, all val sets pooled: recall / precision %", "",
              "Recall: reference frames of the quality output as the same chord (root and quality). Precision: frames "
              "output as the quality that are that chord in the reference. dim includes dim7. Reference seconds: "
              + ", ".join(f"{q} {ref_s[q]['ref_s']:.0f}" for q in qs) + ".", "",
              "| system | point | " + " | ".join(qs) + " |", "|---|---|" + "---:|" * len(qs)]
    for name, by in results.items():
        for point, r in by.items():
            cells = [f"{r['quality'][q]['recall']:.1f} / {r['quality'][q]['precision']:.1f}"
                     if r["quality"][q]["out_s"] else f"{r['quality'][q]['recall']:.1f} / -" for q in qs]
            lines.append(f"| {name} | {point} | " + " | ".join(cells) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "val_table.md").write_text("\n".join(lines) + "\n")
    (OUT / "val_results.json").write_text(json.dumps(results, indent=1))
    print("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["tables", "train", "posteriors", "score"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", type=int, help="replay excerpts per batch (configs/v15.yaml: 5)")
    parser.add_argument("--focus", type=int, help="focus excerpts per batch (configs/v15.yaml: 3)")
    parser.add_argument("--out", type=Path, help="run folder (train)")
    parser.add_argument("--run", type=Path, help="run folder (posteriors)")
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int, help="dry run: override finetune.max_epochs")
    parser.add_argument("--replay-excerpts", type=int, help="dry run: override finetune.replay_excerpts")
    args = parser.parse_args(argv)
    if args.step == "tables":
        build_tables(args)
    elif args.step == "train":
        if args.out is None:
            parser.error("train needs --out")
        train(args)
    elif args.step == "posteriors":
        save_posteriors(args.run, config())
    else:
        score_all(args)


if __name__ == "__main__":
    main()
