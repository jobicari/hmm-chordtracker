"""Train a vocabulary-A CRNN with train_v10's options; the run is decoded by run_v10.py points.

    python scripts/train_v10.py --seed 0 --out results/v10/stop7_seed0 --early-stop sevenths
    python scripts/train_v10.py --seed 0 --out results/v10/heads_seed0 --aux-weight 0.5
    python scripts/train_v10.py --seed 0 --out results/v10/aug_seed0 --gain-db 3 --tilt-db 6 --noise 0.05
    python scripts/train_v10.py --seed 0 --out <dir> --limit 2 --max-epochs 1       # dry run

Data, vocabulary, excerpts and pitch shift are V8's (run_v8.load_song, configs/v10.yaml). The run folder
gets model.pt, prior.json, history.json and run.json (the train options, and whether the model has heads).
Validation only: the val sets drive early stopping, test is never loaded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import acoustic_hmm, data_choco, frame_score, paths, train_v8, train_v10, transitions_v8

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402
import run_v8  # noqa: E402

VOCAB = "A"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v10.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--early-stop", choices=["frame_accuracy", "sevenths"])
    parser.add_argument("--aux-weight", type=float)
    parser.add_argument("--gain-db", type=float)
    parser.add_argument("--tilt-db", type=float)
    parser.add_argument("--noise", type=float)
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args(argv)
    cfg = run_v5.load_config(args.config)
    t = cfg["train"]
    t["seed"] = args.seed
    for name, value in (("early_stop", args.early_stop), ("aux_weight", args.aux_weight),
                        ("max_epochs", args.max_epochs)):
        if value is not None:
            t[name] = value
    for name, value in (("gain_db", args.gain_db), ("tilt_db", args.tilt_db), ("noise", args.noise)):
        if value is not None:
            t.setdefault("augment", {})[name] = value
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    args.out.mkdir(parents=True, exist_ok=True)

    takes = run_v8.train_takes(cfg, first)
    train_songs = [run_v8.load_song(take, cfg, VOCAB) for take in takes]
    val_takes = {**{n: first(ts) for n, ts in run_v5.evaluation_sets("val").items()},
                 "choco": first(data_choco.choco_takes("val", cfg))}
    val_songs = {n: [run_v8.load_song(take, cfg, VOCAB) for take in ts] for n, ts in val_takes.items()}
    sr, hop = cfg["audio"]["sr"], cfg["cqt"]["hop_length"]
    labels = run_v8.class_labels(VOCAB)
    scored = {n: [(song, *frame_score.tables(take.annotation, labels, song.cqt.shape[1], hop / sr, "sevenths"))
                  for take, song in zip(val_takes[n], val_songs[n])] for n in val_takes}
    bass = [train_v10.bass_targets(take.annotation, acoustic_hmm.frame_times(song.cqt.shape[1], sr, hop))
            for take, song in zip(takes, train_songs)] if t.get("aux_weight", 0) else None
    n = run_v8.n_classes(VOCAB)
    print(f"train: {len(train_songs)} songs, seed {args.seed}, early stop {t.get('early_stop')}, "
          f"aux weight {t.get('aux_weight', 0)}, augment {t.get('augment')}", flush=True)
    model, history = train_v10.train(train_songs, val_songs, cfg, n, bass, scored)
    train_v8.save(model, args.out / "model.pt")
    prior = transitions_v8.training_prior(np.concatenate([s.targets for s in train_songs]), n)
    (args.out / "prior.json").write_text(json.dumps(prior.tolist()))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "run.json").write_text(json.dumps({"train": t, "heads": bool(t.get("aux_weight", 0)),
                                                   "n_qualities": n // 12}, indent=2))
    key = "val_mean" if t.get("early_stop", "frame_accuracy") == "frame_accuracy" else "val_sevenths_mean"
    print(f"best epoch {max(history, key=lambda h: h[key])['epoch']} of {len(history)}, "
          f"{sum(h['seconds'] for h in history) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
