"""Train V11's stage B again from V11's stage A model, on the rebuilt cache, with or without the played bass.

    python scripts/train_v13.py --seed 0 --out results/v13/control_seed0                    # (a) V11's stage B
    python scripts/train_v13.py --seed 0 --out results/v13/bass_seed0 --bass-weight 0.5     # (b) + played bass
    python scripts/train_v13.py --seed 0 --out <dir> --limit 2 --max-epochs 1 [--bass-weight 0.5]   # dry run

Stage B as scripts/train_v11.py (configs/v13.yaml = configs/v11.yaml's training): labelled data, chord CE
+ KL to the V8 ensemble, from results/v11/pool_kd1_seed<seed>/stage_a/model.pt (pre-trained on the
unlabelled pool, whose cache already had the 12-bin tuning). The RWC-P and GuitarSet features come from
the rebuilt cache (scripts/v13_cache.py). With --bass-weight > 0 the model is model_v13.ChordCRNNBass and
the loss adds the bass head's CE against the played bass: the lowest note of the bass tracks of RWC-P's
aligned MIDI (midi.bass_notes, train_v13.played_bass); other songs have no bass target. The run folder
gets model.pt, prior.json, history.json and run.json. Validation only: test is never loaded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from chordtracker import data_choco, midi, paths, train_v8, train_v11, train_v13, transitions_v8

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402
import run_v8  # noqa: E402

VOCAB = "A"
MIDI_DIR = paths.DATA_DIR / "rwc" / "rwc-annotations" / "01_annotations_preprocessed" / "MIDI_aligned" / "RWC-P"


def stage_a(seed: int) -> Path:
    return paths.REPO_ROOT / "results" / "v11" / f"pool_kd1_seed{seed}" / "stage_a" / "model.pt"


def bass_of(song, cfg: dict) -> np.ndarray | None:
    """The played-bass targets of an RWC-P song with a MIDI bass part; None for every other song."""
    if not song.key.startswith("rwc/"):
        return None
    notes = midi.bass_notes(midi.read(MIDI_DIR / f"{song.key.split('/')[1]}.mid"))
    if not notes:
        return None
    return train_v13.played_bass(notes, song.cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v13.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bass-weight", type=float)
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args(argv)
    cfg = run_v5.load_config(args.config)
    t = cfg["train"]
    t["seed"] = args.seed
    for name, value in (("bass_weight", args.bass_weight), ("max_epochs", args.max_epochs)):
        if value is not None:
            t[name] = value
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    args.out.mkdir(parents=True, exist_ok=True)

    n = run_v8.n_classes(VOCAB)
    teachers = [run_v8.model_for(paths.REPO_ROOT / p, n, cfg) for p in t["teachers"]]
    train_songs = [run_v8.load_song(take, cfg, VOCAB) for take in run_v8.train_takes(cfg, first)]
    val_takes = {**{name: first(ts) for name, ts in run_v5.evaluation_sets("val").items()},
                 "choco": first(data_choco.choco_takes("val", cfg))}
    val_songs = {name: [run_v8.load_song(take, cfg, VOCAB) for take in ts] for name, ts in val_takes.items()}
    init = torch.load(stage_a(args.seed), map_location="cpu")
    if t["bass_weight"]:
        bass = [bass_of(s, cfg) for s in train_songs]
        with_bass = sum(b is not None for b in bass)
        print(f"train: {len(train_songs)} songs, {with_bass} with a played-bass target, seed {args.seed}, "
              f"bass weight {t['bass_weight']}", flush=True)
        model, history = train_v13.train(train_songs, val_songs, cfg, n, teachers, init, bass)
    else:
        print(f"train: {len(train_songs)} songs, seed {args.seed}, no bass loss (V11's stage B)", flush=True)
        model, history = train_v11.train(train_songs, val_songs, cfg, n, teachers, init)
    train_v8.save(model, args.out / "model.pt")
    prior = transitions_v8.training_prior(np.concatenate([s.targets for s in train_songs]), n)
    (args.out / "prior.json").write_text(json.dumps(prior.tolist()))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "run.json").write_text(json.dumps({"train": t, "stage_a": str(stage_a(args.seed).relative_to(
        paths.REPO_ROOT))}, indent=2))
    print(f"best epoch {max(history, key=lambda h: h['val_mean'])['epoch']} of {len(history)}, "
          f"{sum(h['seconds'] for h in history) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
