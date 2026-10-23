"""V5's baseline: the V4 system refitted on V5's training data, scored on V5's two val sets.

    python scripts/run_v5_baseline.py

Same as V4 in every way (V2 treble + bass beat chroma, logistic regression C 10 on 12 transpositions,
division by the training prior, Billboard change prior tau 1, p_self swept) except the training data:
RWC-P train (60 songs) + GuitarSet comp train (60 takes), as the V5 CNN. So the comparison with V5
isolates the acoustic model. Val only: RWC-P val and GuitarSet val, scored separately, written to
results/v5/baseline_v4_refit/val/. Never touches a test split.
"""
import json

import librosa
import numpy as np
import yaml

from chordtracker import beats, data_v5, evaluate, experiment, features_v2, learned_emission as le, paths, transitions

OUT = paths.REPO_ROOT / "results" / "v5" / "baseline_v4_refit"


def load_config() -> dict:
    """configs/v4.yaml on top of its base, exactly as run_v4."""
    cfg = yaml.safe_load((paths.REPO_ROOT / "configs" / "v4.yaml").read_text())
    return {**yaml.safe_load((paths.REPO_ROOT / "configs" / cfg["base_config"]).read_text()), **cfg}


def change_prior(cfg: dict) -> np.ndarray:
    """(24, 24) Billboard Q, as run_v4."""
    t = cfg["transitions"]
    counts = np.array(json.loads((paths.REPO_ROOT / t["counts"]).read_text())["counts"], dtype=float)
    Q = transitions.change_matrix(counts, alpha=t["alpha"], tau=t["tau"])[:le.N_CLASSES, :le.N_CLASSES]
    return Q / Q.sum(axis=1, keepdims=True)


def take_beats(take: data_v5.Take, cfg: dict) -> le.BeatFeatures:
    """le.take_beats for any data_v5.Take (RWC-P has a csv, not a .lab): same features, beats and edges."""
    treble, bass = features_v2.extract(take.audio, cfg)
    bounds = beats.segment_bounds(beats.beat_frames(take.audio, cfg), treble.shape[1])
    duration = float(take.annotation.intervals[-1, 1])
    edges = librosa.frames_to_time(bounds, sr=cfg["audio"]["sr"], hop_length=cfg["features"]["hop_length"])
    edges[0], edges[-1] = 0.0, duration
    agg = cfg["beats"]["aggregate"]
    features = le.beat_features(beats.beat_sync(treble, bounds, agg), beats.beat_sync(bass, bounds, agg))
    return le.BeatFeatures(take.key, features, edges, take.annotation, duration)


def main() -> None:
    cfg = load_config()
    train = [take_beats(t, cfg) for t in data_v5.rwc_takes("train") + data_v5.guitarset_takes("train")]
    model = le.fit(*le.training_rows(train), C=cfg["emission"]["C"])
    le.save(model, OUT / "model.json")
    print(f"fitted on {len(train)} train songs", flush=True)

    Q, summary_rows, per_song_rows = change_prior(cfg), [], []
    grid = cfg["decoder"]["p_self_grid"]
    for dataset, takes in {"rwc": data_v5.rwc_takes("val"), "guitarset": data_v5.guitarset_takes("val")}.items():
        evidences = [le.to_evidence(take_beats(t, cfg), model) for t in takes]
        for p_self in grid:
            A = transitions.transition(Q, p_self)
            scores = []
            for ev in evidences:
                path = librosa.sequence.viterbi_discriminative(ev.posteriors, A, p_state=model["prior"])
                song_score, cpm = beats.score_path(path, le.LABELS, ev)
                scores.append(song_score)
                per_song_rows.append({"dataset": dataset, "system": p_self, "song_id": ev.song_id,
                                      "wcsr": song_score.wcsr, "changes_per_min": cpm})
            summary_rows.append({"dataset": dataset, "system": p_self, "wcsr": evaluate.pooled(scores).wcsr,
                                 "n_takes": len(evidences)})
    mean_rows = [{"dataset": "mean", "system": p,
                  "wcsr": float(np.mean([r["wcsr"] for r in summary_rows if r["system"] == p]))} for p in grid]
    summary = {"system": "v4 refit on v5 train", "split": "val", "rows": summary_rows + mean_rows,
               "chosen_p_self": experiment.choose_p_self(mean_rows)}
    experiment.write_results(summary, per_song_rows, OUT / "val")
    for row in summary_rows + mean_rows:
        print(f"  {row['dataset']:>9}  p_self {row['system']:.2f}  WCSR {100 * row['wcsr']:6.2f}%")
    print(f"chosen p_self (mean of the two val sets): {summary['chosen_p_self']}")


if __name__ == "__main__":
    main()
