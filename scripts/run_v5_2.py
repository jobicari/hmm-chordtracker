"""Run V5.2 (V5 with weight decay; AAM from step 2) on RWC-P and GuitarSet.

    python scripts/run_v5_2.py --split val --train                                  # configs/v5_2.yaml as is
    python scripts/run_v5_2.py --split val --train --weight-decay 1e-3 --out results/v5.2/step1/wd_1e-3
    python scripts/run_v5_2.py --split val --train --seed 1 --out results/v5.2/step1/v5_seed1
    python scripts/run_v5_2.py --split val --train --aam-ratio 1 --out results/v5.2/step2/aam1000_seed0
    python scripts/run_v5_2.py --split val --train --aam-ratio 1 --aam-songs 250 --out results/v5.2/step2/aam250_seed0
    python scripts/run_v5_2.py --split val --train --limit 2 --max-epochs 1 --out <dir>   # dry run

--weight-decay, --seed and --aam-ratio override cfg["train"] for one run, so each step's trainings are
separate calls with their own --out. With an AAM ratio, the AAM songs are loaded from the log-CQT
cache (data_aam) and only train on them; val and test stay RWC-P and GuitarSet. The prior the decoder
divides by is the expected class share of one epoch: real and AAM shares weighted 1 : ratio.
The data, beats, decoder and sweep are run_v5's, imported unchanged; only the training differs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import acoustic_hmm, data_aam, data_v5, experiment, paths, train_v5, train_v5_2

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (parse_args, load_config, evaluation_sets, evidence, sweep, change_prior)


def split_overrides(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Take --weight-decay, --seed, --aam-ratio and --aam-songs out of argv; the rest goes to run_v5.parse_args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aam-ratio", type=float, help="AAM frames per real frame each epoch; -1 = all")
    parser.add_argument("--aam-songs", type=int, help="use only the first N cached AAM songs")
    return parser.parse_known_args(argv)


def mixed_prior(train_songs: list, aam_songs: list, ratio: float) -> np.ndarray:
    """Expected class share of one epoch: the real prior and the AAM prior weighted 1 : ratio.

    With no AAM this is V5's prior. ratio -1 (all AAM every epoch) weights them by frame counts.
    """
    real = np.concatenate([s.targets for s in train_songs])
    if not ratio or not aam_songs:
        return acoustic_hmm.training_prior(real)
    aam = np.concatenate([s.targets for s in aam_songs])
    if ratio < 0:
        return acoustic_hmm.training_prior(np.concatenate([real, aam]))
    w = min(ratio, (aam >= 0).sum() / (real >= 0).sum())
    return (acoustic_hmm.training_prior(real) + w * acoustic_hmm.training_prior(aam)) / (1 + w)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides, argv = split_overrides(argv)
    if "--config" not in argv:
        argv += ["--config", str(paths.REPO_ROOT / "configs" / "v5_2.yaml")]
    args = run_v5.parse_args(argv)
    cfg = run_v5.load_config(args.config)
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    if overrides.weight_decay is not None:
        cfg["train"]["weight_decay"] = overrides.weight_decay
    if overrides.seed is not None:
        cfg["train"]["seed"] = overrides.seed
    if overrides.aam_ratio is not None:
        cfg["train"]["aam_ratio"] = overrides.aam_ratio
    ratio = cfg["train"].get("aam_ratio", 0)
    out = Path(args.out) if args.out else paths.REPO_ROOT / "results" / cfg["version"]
    out.mkdir(parents=True, exist_ok=True)
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)

    if args.train:
        train_takes = first(data_v5.rwc_takes("train")) + first(data_v5.guitarset_takes("train"))
        train_songs = [data_v5.load(t, cfg) for t in train_takes]
        val_songs = {name: [data_v5.load(t, cfg) for t in first(takes)]
                     for name, takes in run_v5.evaluation_sets("val").items()}
        aam_ids = first(data_aam.aam_song_ids())[:overrides.aam_songs]
        aam_songs = [data_v5.load(t, cfg) for t in data_aam.aam_takes(aam_ids)] if ratio else []
        print(f"train: {len(train_songs)} songs + {len(aam_songs)} AAM (ratio {ratio}), "
              f"weight decay {cfg['train']['weight_decay']}, seed {cfg['train']['seed']}", flush=True)
        model, history = train_v5_2.train(train_songs, val_songs, cfg, aam_songs)
        train_v5.save(model, out / "model.pt")
        (out / "prior.json").write_text(json.dumps(mixed_prior(train_songs, aam_songs, ratio).tolist()))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"best epoch {max(history, key=lambda h: h['val_mean'])['epoch']} of {len(history)} -> {out / 'model.pt'}")
    model = train_v5.load(out / "model.pt", cfg["train"]["dropout"])
    prior = np.array(json.loads((out / "prior.json").read_text()))

    Q = run_v5.change_prior(cfg)
    grid = cfg["decoder"]["p_self_grid"] if args.split == "val" else [args.p_self]
    summary_rows, per_song_rows = [], []
    for dataset, takes in run_v5.evaluation_sets(args.split).items():
        evidences = [run_v5.evidence(model, t, cfg) for t in first(takes)]
        rows, songs = run_v5.sweep(dataset, evidences, Q, prior, grid)
        summary_rows += rows
        per_song_rows += songs
    mean_rows = [{"dataset": "mean", "system": p,
                  "wcsr": float(np.mean([r["wcsr"] for r in summary_rows if r["system"] == p]))} for p in grid]
    summary = {"version": cfg["version"], "split": args.split, "config": cfg, "rows": summary_rows + mean_rows}
    if args.split == "val":
        summary["chosen_p_self"] = experiment.choose_p_self(mean_rows)
    experiment.write_results(summary, per_song_rows, out / args.split)

    for row in summary_rows + mean_rows:
        print(f"  {row['dataset']:>9}  p_self {row['system']:.2f}  WCSR {100 * row['wcsr']:6.2f}%")
    if "chosen_p_self" in summary:
        print(f"chosen p_self (mean of the two val sets): {summary['chosen_p_self']}")


if __name__ == "__main__":
    main()
