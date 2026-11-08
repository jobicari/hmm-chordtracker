"""Train a vocabulary-A CRNN distilled from the V8 ensemble (train_v11); decode it with run_v10.py points.

    python scripts/train_v11.py --seed 0 --out results/v11/kd1_seed0
    python scripts/run_v10.py points --run results/v11/kd1_seed0 --name v11_points_kd1_seed0
    python scripts/train_v11.py --seed 0 --out results/v11/pool_kd1_seed0 --pool       # stage A on the pool, then B
    python scripts/train_v11.py --seed 0 --out <dir> --limit 2 --max-epochs 1 [--pool]  # dry run

With --pool, stage A pre-trains on the unlabelled pool (scripts/v11_pool.py's manifest and cache) with the
KL term alone, train.pool_ratio times as many excerpts an epoch as the labelled data has; its best model
and history go to <out>/stage_a/. Stage B is then the plain run started from that model.

Data, vocabulary, excerpts and pitch shift are V8's (run_v8.load_song, configs/v11.yaml); the teachers are
the saved runs in train.teachers. The run folder gets model.pt, prior.json (the real training labels, as
V8), history.json and run.json. Validation only: the val sets drive early stopping, test is never loaded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import data_choco, data_v6, paths, train_v8, train_v11, transitions_v8
from chordtracker.data_v5 import Song

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402
import run_v8  # noqa: E402
import v11_pool  # noqa: E402

VOCAB = "A"


def load_pool(first) -> list[Song]:
    """Every cached pool recording as a Song with no targets; the features stay float16 in memory."""
    rows = first(v11_pool.pool_rows())
    missing = [r["id"] for r in rows if not (v11_pool.CACHE / f"{r['id']}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} pool recordings not cached (scripts/v11_pool.py cache)")
    songs = []
    for r in rows:
        cqt = np.load(v11_pool.CACHE / f"{r['id']}.npy")
        songs.append(Song(r["id"], cqt, np.full(cqt.shape[1], -1)))
    return songs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v11.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kl-weight", type=float)
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--pool", action="store_true", help="pre-train on the unlabelled pool first (stage A)")
    args = parser.parse_args(argv)
    cfg = run_v5.load_config(args.config)
    t = cfg["train"]
    t["seed"] = args.seed
    for name, value in (("kl_weight", args.kl_weight), ("max_epochs", args.max_epochs)):
        if value is not None:
            t[name] = value
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    args.out.mkdir(parents=True, exist_ok=True)

    n = run_v8.n_classes(VOCAB)
    teachers = [run_v8.model_for(paths.REPO_ROOT / p, n, cfg) for p in t["teachers"]] if t.get("kl_weight", 0) else None
    train_songs = [run_v8.load_song(take, cfg, VOCAB) for take in run_v8.train_takes(cfg, first)]
    val_takes = {**{name: first(ts) for name, ts in run_v5.evaluation_sets("val").items()},
                 "choco": first(data_choco.choco_takes("val", cfg))}
    val_songs = {name: [run_v8.load_song(take, cfg, VOCAB) for take in ts] for name, ts in val_takes.items()}
    print(f"train: {len(train_songs)} songs, seed {args.seed}, kl weight {t.get('kl_weight', 0)}, "
          f"teachers {t['teachers'] if teachers else None}", flush=True)
    init = None
    if args.pool:
        if not teachers:
            parser.error("--pool needs kl_weight > 0 (the teachers)")
        pool = load_pool(first)
        n_labelled = len(data_v6.excerpt_starts(train_songs, t["excerpt_frames"], np.random.default_rng(0)))
        n_excerpts = int(t["pool_ratio"] * n_labelled)
        print(f"stage A: {len(pool)} pool recordings, {n_excerpts} excerpts an epoch "
              f"({t['pool_ratio']} x {n_labelled} labelled)", flush=True)
        stage_a, history_a = train_v11.train_pool(pool, val_songs, cfg, n, teachers, n_excerpts)
        (args.out / "stage_a").mkdir(exist_ok=True)
        train_v8.save(stage_a, args.out / "stage_a" / "model.pt")
        (args.out / "stage_a" / "history.json").write_text(json.dumps(history_a, indent=2))
        init = {k: v.detach().cpu().clone() for k, v in stage_a.state_dict().items()}
        del pool
        print("stage B: labelled data from stage A's best model", flush=True)
    model, history = train_v11.train(train_songs, val_songs, cfg, n, teachers, init)
    train_v8.save(model, args.out / "model.pt")
    prior = transitions_v8.training_prior(np.concatenate([s.targets for s in train_songs]), n)
    (args.out / "prior.json").write_text(json.dumps(prior.tolist()))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "run.json").write_text(json.dumps({"train": t, "pool": args.pool}, indent=2))
    print(f"best epoch {max(history, key=lambda h: h['val_mean'])['epoch']} of {len(history)}, "
          f"{sum(h['seconds'] for h in history) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
