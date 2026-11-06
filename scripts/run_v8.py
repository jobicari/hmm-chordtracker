"""Run V8: train the causal CRNN (C2 grid, input look-ahead 2) on RWC-P, GuitarSet and the ChoCo export
with a richer vocabulary, then decode and score on validation.

    python scripts/run_v8.py --vocab A --train --seed 0          # -> results/v8/A_seed0
    python scripts/run_v8.py --vocab majmin --train --seed 0     # V7.3's 24 classes on the new data
    python scripts/run_v8.py --vocab majmin --from-run results/v7_2/c2_la2_seed0 --out results/v8/v7_3_as_is
    python scripts/run_v8.py --vocab A --train --limit 2 --max-epochs 1 --out <dir>   # dry run
    python scripts/run_v8.py --vocab A --seed 0 --transitions train   # decode results/v8/A_seed0 again

Vocabularies: majmin (V7.3's targets, data_v5.target), A, B, B9 (data_v8). Decoders are V7's
(frame-argmax, forward-filter, fixed-lag-viterbi at every lag, offline-frame-viterbi) with the prior
and transition matrix of the vocabulary. transitions.source (or --transitions) chooses the matrix:
billboard (transitions_v8.change_matrix; run_v5.change_prior for majmin) or train (moves counted in
the training targets, transitions_v8.label_change_matrix, counts kept in the run directory). Results go
to <run>/val for billboard and <run>/val_<source>_tau<tau> otherwise.
Scores: aligned WCSR with mir_eval majmin and sevenths, pooled per set (rwc, guitarset, choco; ChoCo
also per partition), never pooled across sets; plus a quality confusion matrix of the frames.
Sevenths reads B's maj6 as maj, so A, B and B9 are scored on the same chords. Test is not run here.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

from chordtracker import (acoustic_hmm, data_choco, data_v5, data_v7_2, data_v8, evaluate, experiment, online_hmm,
                          paths, train_v7_2, train_v8, transitions, transitions_v8)
from chordtracker.learned_emission import LABELS        # C:maj ... B:maj, C:min ... B:min
from chordtracker.model_v5 import device
from chordtracker.model_v8 import ChordCRNN

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config, evaluation_sets, change_prior)
import run_v7  # noqa: E402  (system names)

METRICS = ("majmin", "sevenths")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v8.yaml")
    parser.add_argument("--vocab", choices=["majmin", *data_v8.VOCABULARIES], required=True)
    parser.add_argument("--train", action="store_true", help="train the CRNN; else load it from --out or --from-run")
    parser.add_argument("--from-run", type=Path, help="score a saved 24-class run (model.pt, prior.json) as it is")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int, help="dry run: override train.max_epochs")
    parser.add_argument("--out", type=Path, help="run directory; default results/v8/<vocab>_seed<seed>")
    parser.add_argument("--transitions", choices=["billboard", "train"], help="override transitions.source")
    parser.add_argument("--tau", type=float, help="override transitions.tau")
    args = parser.parse_args(argv)
    if args.from_run and (args.train or args.vocab != "majmin"):
        parser.error("--from-run scores a saved 24-class model: use --vocab majmin and no --train")
    return args


def n_classes(vocab: str) -> int:
    return 24 if vocab == "majmin" else 12 * len(data_v8.VOCABULARIES[vocab])


def class_labels(vocab: str) -> list[str]:
    """The label of every class as scored: B's maj6 as maj, so every vocabulary is scored on the same chords."""
    if vocab == "majmin":
        return list(LABELS)
    return [name.replace(":maj6", ":maj") for name in data_v8.class_names(vocab)]


def qualities(vocab: str) -> list[str]:
    return ["maj", "min"] if vocab == "majmin" else list(data_v8.VOCABULARIES[vocab])


def load_song(take, cfg: dict, vocab: str):
    """Cached causal log-CQT (data_choco for ChoCo, data_v7_2 otherwise) with the vocabulary's targets."""
    song = data_choco.load(take, cfg) if take.key.startswith("choco/") else data_v7_2.load(take, cfg)
    return song if vocab == "majmin" else data_v8.retarget(song, take.annotation, cfg, vocab)


def decoder_parts(cfg: dict, vocab: str, prior: np.ndarray, label_counts: np.ndarray | None = None) -> np.ndarray:
    """Q for the vocabulary: from the Billboard counts, or from label_counts when transitions.source is train."""
    t = cfg["transitions"]
    if t.get("source", "billboard") == "train":
        return transitions_v8.label_change_matrix(label_counts, alpha=t["alpha"], tau=t["tau"])
    if vocab == "majmin":
        return run_v5.change_prior(cfg)
    counts = np.array(json.loads((paths.REPO_ROOT / t["counts"]).read_text())["counts"], dtype=float)
    return transitions_v8.change_matrix(counts, vocab, prior, alpha=t["alpha"], tau=t["tau"])


def train_takes(cfg: dict, first) -> list:
    return (first(data_v5.rwc_takes("train")) + first(data_v5.guitarset_takes("train"))
            + first(data_choco.choco_takes("train", cfg)))


def label_counts(run: Path, cfg: dict, vocab: str, first) -> np.ndarray:
    """Changes in the training targets (transitions_v8.label_counts), kept in run/label_counts_<vocab>.json."""
    path = run / f"label_counts_{vocab}.json"
    if not path.exists():
        targets = [load_song(t, cfg, vocab).targets for t in train_takes(cfg, first)]
        counts = transitions_v8.label_counts(targets, len(qualities(vocab)))
        path.write_text(json.dumps({"qualities": qualities(vocab), "counts": counts.tolist()}))
    return np.array(json.loads(path.read_text())["counts"])


def results_dir(cfg: dict) -> str:
    t = cfg["transitions"]
    source = t.get("source", "billboard")
    return "val" if source == "billboard" else f"val_{source}_tau{t['tau']:g}"


def frame_paths(post: np.ndarray, Q: np.ndarray, prior: np.ndarray, d: dict):
    """Every (system, p_self, lag, path) of one song, as run_v7.frame_paths for the V7 frame decoders."""
    yield run_v7.ARGMAX, None, 0, post.argmax(axis=0)
    for p_self in d["p_self_grid"]:
        A = transitions.transition(Q, p_self)
        yield run_v7.FILTER, p_self, 0, online_hmm.forward_filter(post, A, prior)
        for lag, path in online_hmm.fixed_lag_viterbi(post, A, prior, d["lags"]).items():
            yield run_v7.FIXED_LAG, p_self, lag, path
        yield run_v7.OFFLINE, p_self, None, librosa.sequence.viterbi_discriminative(post, A, p_state=prior)


def confusion(targets: np.ndarray, path: np.ndarray, n_q: int) -> np.ndarray:
    """(n_q, n_q) frames by (true quality, found quality), frames with a target only; the root is ignored."""
    keep = targets >= 0
    return np.bincount((targets[keep] // 12) * n_q + path[keep] // 12, minlength=n_q * n_q).reshape(n_q, n_q)


def score_song(post: np.ndarray, song, take, cfg: dict, vocab: str, Q, prior, dataset: str, group: str):
    """Per-song rows for every decoder setting, and the quality confusion of each setting."""
    times = acoustic_hmm.frame_times(post.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    duration = float(take.annotation.intervals[-1, 1])
    labels, n_q = class_labels(vocab), len(qualities(vocab))
    rows, confusions = [], {}
    for system, p_self, lag, path in frame_paths(post, Q, prior, cfg["decoder"]):
        estimate = evaluate.path_to_annotation(path, labels, times, duration)
        row = {"dataset": dataset, "group": group, "system": system, "p_self": p_self, "lag": lag,
               "song_id": take.key, "changes_per_min": evaluate.changes_per_minute(estimate, duration)}
        for metric in METRICS:
            s = evaluate.score(take.annotation, estimate, metric)
            row[f"{metric}_scored_s"], row[f"{metric}_matched_s"] = s.scored_s, s.matched_s
        rows.append(row)
        confusions[(system, p_self, lag)] = confusion(song.targets, path, n_q)
    return rows, confusions


def summarize(per_song: list[dict], key: str = "dataset") -> list[dict]:
    """Pooled WCSR per (set, system, p_self, lag) for both metrics, then the mean over rwc, guitarset, choco."""
    settings = list(dict.fromkeys((r[key], r["system"], r["p_self"], r["lag"]) for r in per_song))
    rows = []
    for name, system, p_self, lag in settings:
        songs = [r for r in per_song if (r[key], r["system"], r["p_self"], r["lag"]) == (name, system, p_self, lag)]
        row = {"dataset": name, "system": system, "p_self": p_self, "lag": lag, "n_takes": len(songs),
               "mean_changes_per_min": float(np.mean([r["changes_per_min"] for r in songs]))}
        for metric in METRICS:
            scored = sum(r[f"{metric}_scored_s"] for r in songs)
            row[metric] = sum(r[f"{metric}_matched_s"] for r in songs) / scored if scored else None
        rows.append(row)
    if key == "dataset":
        for system, p_self, lag in dict.fromkeys(k[1:] for k in settings):
            sets = [r for r in rows if (r["system"], r["p_self"], r["lag"]) == (system, p_self, lag)]
            rows.append({"dataset": "mean", "system": system, "p_self": p_self, "lag": lag,
                         **{m: float(np.mean([r[m] for r in sets if r[m] is not None])) for m in METRICS}})
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = run_v5.load_config(args.config)
    cfg["train"]["seed"] = args.seed
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.transitions:
        cfg["transitions"]["source"] = args.transitions
    if args.tau is not None:
        cfg["transitions"]["tau"] = args.tau
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    out = args.out or paths.REPO_ROOT / "results" / "v8" / f"{args.vocab}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    val_takes = {**{n: first(t) for n, t in run_v5.evaluation_sets("val").items()},
                 "choco": first(data_choco.choco_takes("val", cfg))}

    if args.train:
        train_songs = [load_song(t, cfg, args.vocab) for t in train_takes(cfg, first)]
        val_songs = {n: [load_song(t, cfg, args.vocab) for t in ts] for n, ts in val_takes.items()}
        print(f"train: {len(train_songs)} songs, vocab {args.vocab} ({n_classes(args.vocab)} classes), "
              f"seed {args.seed}", flush=True)
        model, history = train_v8.train(train_songs, val_songs, cfg, n_classes(args.vocab))
        train_v8.save(model, out / "model.pt")
        prior = transitions_v8.training_prior(np.concatenate([s.targets for s in train_songs]), n_classes(args.vocab))
        (out / "prior.json").write_text(json.dumps(prior.tolist()))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"best epoch {max(history, key=lambda h: h['val_mean'])['epoch']} of {len(history)}, "
              f"{sum(h['seconds'] for h in history) / 60:.0f} min", flush=True)

    run = args.from_run or out
    if args.vocab == "majmin":
        model = train_v7_2.load(run / "model.pt", False, cfg["train"]["dropout"])
    else:
        model = model_for(run, n_classes(args.vocab), cfg)
    prior = np.array(json.loads((run / "prior.json").read_text()))
    counts = label_counts(out, cfg, args.vocab, first) if cfg["transitions"].get("source") == "train" else None
    Q = decoder_parts(cfg, args.vocab, prior, counts)
    per_song, confusions = [], {}
    for dataset, ts in val_takes.items():
        for take in ts:
            song = load_song(take, cfg, args.vocab)
            post = train_v7_2.song_posteriors(model, song, cfg["train"]["input_lookahead"])
            group = take.key.split("/")[1].rsplit("_", 1)[0] if dataset == "choco" else dataset
            rows, conf = score_song(post, song, take, cfg, args.vocab, Q, prior, dataset, group)
            per_song += rows
            for k, c in conf.items():
                confusions[(dataset, *k)] = confusions.get((dataset, *k), 0) + c
        print(f"{dataset}: {len(ts)} takes", flush=True)
    rows = summarize(per_song)
    groups = summarize([r for r in per_song if r["dataset"] == "choco"], key="group")
    summary = {"version": cfg["version"], "split": "val", "vocab": args.vocab, "qualities": qualities(args.vocab),
               "config": cfg, "rows": rows, "choco_partitions": groups,
               "confusion": [{"dataset": k[0], "system": k[1], "p_self": k[2], "lag": k[3], "frames": v.tolist()}
                             for k, v in confusions.items()]}
    experiment.write_results(summary, per_song, out / results_dir(cfg))
    for r in rows:
        if r["system"] == run_v7.ARGMAX:
            print(f"  {r['dataset']:9s} network alone: majmin {100 * r['majmin']:.2f}%  sevenths {100 * r['sevenths']:.2f}%")


def model_for(run: Path, n: int, cfg: dict) -> ChordCRNN:
    """A saved V8 model with n classes, on device(), in eval mode."""
    model = ChordCRNN(n, bidirectional=False, dropout=cfg["train"]["dropout"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model.to(device()).eval()


if __name__ == "__main__":
    main()
