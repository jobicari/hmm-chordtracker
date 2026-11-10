"""Run V5 (CNN acoustic model on the log-CQT, V4's beat decoder) on RWC-P and GuitarSet.

    python scripts/run_v5.py --split val --train          # train once, save the model, sweep p_self on val
    python scripts/run_v5.py --split val                  # reuse the saved model, sweep p_self again
    python scripts/run_v5.py --split test --p-self 0.5    # once, with the saved model and the p_self chosen on val
    python scripts/run_v5.py --split val --train --limit 2 --max-epochs 1 --out <dir>   # dry run

Training data: RWC-P train (60 songs) + GuitarSet comp train (60 takes). The two val sets are scored
separately and p_self is chosen on the mean of their WCSRs. The model, the decoder's prior and the
training history are saved to results/v5/; each split's results go to results/v5/<split>/ as in V4.
The first run fills the log-CQT cache (data/processed/v5_logcqt/, ~13 min); later runs read it.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import yaml

from chordtracker import acoustic_hmm, beats, data_v5, evaluate, experiment, paths, train_v5, transitions
from chordtracker.data.data import load_split
from chordtracker.learned_emission import LABELS        # C:maj ... B:maj, C:min ... B:min, as V4 and data_v5.target


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v5.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--train", action="store_true", help="train the CNN (val only); else load the saved one")
    parser.add_argument("--p-self", type=float, help="the p_self chosen on val; required with --split test")
    parser.add_argument("--limit", type=int, help="dry run: only the first N songs of every set")
    parser.add_argument("--max-epochs", type=int, help="dry run: override train.max_epochs")
    parser.add_argument("--out", type=Path, help="write everything here instead of results/v5 (dry run)")
    args = parser.parse_args(argv)
    if args.split == "test" and (args.p_self is None or args.train):
        parser.error("--split test needs --p-self and never trains")
    if args.split == "val" and args.p_self is not None:
        parser.error("--split val sweeps the grid from the config; --p-self is only for test")
    return args


def load_config(path: Path) -> dict:
    """A config on top of its base (base_config, e.g. v2.yaml), and the base on its own if it has one
    (v15.yaml on v14.yaml on v2.yaml): each config's sections replace its base's."""
    cfg = yaml.safe_load(Path(path).read_text())
    if "base_config" not in cfg:
        return cfg
    return {**load_config(Path(path).parent / cfg["base_config"]), **cfg}


def change_prior(cfg: dict) -> np.ndarray:
    """(24, 24) Q from the Billboard counts, N dropped and the rows renormalized (as run_v4)."""
    t, n = cfg["transitions"], acoustic_hmm.N_CLASSES
    counts = np.array(json.loads((paths.REPO_ROOT / t["counts"]).read_text())["counts"], dtype=float)
    Q = transitions.change_matrix(counts, alpha=t["alpha"], tau=t["tau"])[:n, :n]
    return Q / Q.sum(axis=1, keepdims=True)


def rwc_test_takes() -> list[data_v5.Take]:
    """The 20 RWC-P test songs. Only here, only for the single test run of the frozen V5."""
    return [data_v5.Take(f"rwc/{s}", data_v5.RWC_DIR / "RWC-P" / f"{s}.wav", data_v5.rwc_annotation(s))
            for s in load_split(data_v5.RWC_SPLIT)["test"]]


def evaluation_sets(split: str) -> dict[str, list[data_v5.Take]]:
    """The takes scored for a split, by dataset; RWC-P and GuitarSet are never pooled."""
    rwc = rwc_test_takes() if split == "test" else data_v5.rwc_takes(split)
    return {"rwc": rwc, "guitarset": data_v5.guitarset_takes(split)}


def beat_edges(take: data_v5.Take, cfg: dict) -> tuple[np.ndarray, float]:
    """V1 beat segments in seconds, as V4: edges[0] = 0 and edges[-1] = end of the reference."""
    duration = float(take.annotation.intervals[-1, 1])
    hop, sr = cfg["features"]["hop_length"], cfg["audio"]["sr"]
    bounds = beats.segment_bounds(beats.beat_frames(take.audio, cfg), int(duration * sr / hop) + 1)
    edges = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)
    edges[0], edges[-1] = 0.0, duration
    return edges, duration


def evidence(model, take: data_v5.Take, cfg: dict) -> beats.BeatEvidence:
    """CNN frame posteriors of one take, pooled into its beats: what the V4 decoder and scoring read."""
    song = data_v5.load(take, cfg)
    frame_post = train_v5.song_posteriors(model, song)
    times = acoustic_hmm.frame_times(frame_post.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    edges, duration = beat_edges(take, cfg)
    return beats.BeatEvidence(take.key, acoustic_hmm.frames_to_beats(frame_post, times, edges), edges,
                              take.annotation, duration)


def sweep(dataset: str, takes: list[beats.BeatEvidence], Q: np.ndarray, prior: np.ndarray, grid: list[float]):
    """Decode every take at every p_self; summary rows (one per p_self) and per-song rows, tagged by dataset."""
    summary_rows, per_song_rows = [], []
    for p_self in grid:
        scores, changes = [], []
        for ev in takes:
            path = acoustic_hmm.decode(ev.posteriors, Q, p_self, prior)
            song_score, changes_per_minute = beats.score_path(path, LABELS, ev)
            scores.append(song_score)
            changes.append(changes_per_minute)
            per_song_rows.append({"dataset": dataset, "system": p_self, "song_id": ev.song_id,
                                  "wcsr": song_score.wcsr, "changes_per_min": changes_per_minute,
                                  "total_s": song_score.total_s, "scored_s": song_score.scored_s,
                                  "matched_s": song_score.matched_s, "reduced_s": song_score.reduced_s,
                                  "excluded_s": song_score.excluded_s})
        total = evaluate.pooled(scores)
        summary_rows.append({"dataset": dataset, "system": p_self, "wcsr": total.wcsr,
                             "mean_changes_per_min": float(np.mean(changes)), "scored_s": total.scored_s,
                             "reduced_s": total.reduced_s, "excluded_s": total.excluded_s, "n_takes": len(takes)})
    return summary_rows, per_song_rows


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    out = args.out or paths.REPO_ROOT / "results" / cfg["version"]
    out.mkdir(parents=True, exist_ok=True)
    first = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)

    if args.train:
        train_takes = first(data_v5.rwc_takes("train")) + first(data_v5.guitarset_takes("train"))
        train_songs = [data_v5.load(t, cfg) for t in train_takes]
        val_songs = {name: [data_v5.load(t, cfg) for t in first(takes)] for name, takes in evaluation_sets("val").items()}
        n_frames = sum(int((s.targets >= 0).sum()) for s in train_songs)
        print(f"train: {len(train_songs)} songs, {n_frames} frames with a target", flush=True)
        model, history = train_v5.train(train_songs, val_songs, cfg)
        train_v5.save(model, out / "model.pt")
        prior = acoustic_hmm.training_prior(np.concatenate([s.targets for s in train_songs]))
        (out / "prior.json").write_text(json.dumps(prior.tolist()))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"best epoch {max(history, key=lambda h: h['val_mean'])['epoch']} of {len(history)} -> {out / 'model.pt'}")
    model = train_v5.load(out / "model.pt", cfg["train"]["dropout"])
    prior = np.array(json.loads((out / "prior.json").read_text()))

    Q = change_prior(cfg)
    grid = cfg["decoder"]["p_self_grid"] if args.split == "val" else [args.p_self]
    summary_rows, per_song_rows = [], []
    for dataset, takes in evaluation_sets(args.split).items():
        evidences = [evidence(model, t, cfg) for t in first(takes)]
        rows, songs = sweep(dataset, evidences, Q, prior, grid)
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
