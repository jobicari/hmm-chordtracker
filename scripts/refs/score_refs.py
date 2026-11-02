"""Score external, pretrained chord recognisers on V6's songs, splits and metric. Nothing is trained or tuned.

    python scripts/refs/score_refs.py --tools btc madmom chordino --splits val
    python scripts/refs/score_refs.py --tools btc madmom chordino --splits test
    python scripts/refs/score_refs.py --summary-only                  # rebuild summary.json from the CSV

Repo code is imported read-only, so the numbers are V6's: run_v5.evaluation_sets gives the songs (same
audio files, same references), evaluate.score and evaluate.pooled give the WCSR. Results go to results/refs/.

The tools stay outside the repo, in CHORD_REFS_DIR (default ~/chordtracker_v5_scratch/refs): btc/ is the BTC
clone, plugins/ holds nnls-chroma.dylib. Every tool's raw output goes there too, to
out/labs/<tool>/<split>/<dataset>/<song>.lab; an existing .lab is reused, so a run can resume.
results/refs/README.md has the environment and the versions.

Label mapping: a raw label becomes Harte, then data_v5.target reads its third as V5/V6 do
(C:7, C:maj7 -> C:maj; A:min7, C:dim -> C:min). N stays N; a chord with no third counts as N.

Pairs: BTC and Chordino on both datasets; madmom on GuitarSet only (its models were trained on RWC Pop).
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True      # nothing is written under the repo

import numpy as np                  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REFS = Path(os.environ.get("CHORD_REFS_DIR", Path.home() / "chordtracker_v5_scratch" / "refs"))
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from chordtracker import data_v5, evaluate                  # noqa: E402
from chordtracker.data.data import ChordAnnotation          # noqa: E402
from chordtracker.learned_emission import LABELS            # noqa: E402
import run_v5                                               # noqa: E402

BTC_DIR = REFS / "btc"
LABS = REFS / "out" / "labs"
RESULTS = REPO / "results" / "refs"
PER_SONG = RESULTS / "per_song.csv"
PAIRS = {"btc": ("rwc", "guitarset"), "chordino": ("rwc", "guitarset"), "madmom": ("guitarset",)}
FIELDS = ["tool", "split", "dataset", "song_id", "wcsr", "total_s", "scored_s", "matched_s", "reduced_s",
          "excluded_s", "n_segments", "no_third_s", "seconds", "status"]
V6_RUN = REPO / "results" / "v6" / "bidir_seed1"
V6_P_SELF = 0.4


# ==================================== tools: audio path -> (intervals, raw labels) ====================================

def btc_runner():
    """BTC's shipped maj-min model, inference as in its test.py (10 s chunks of 108 frames, no overlap)."""
    import torch
    import yaml
    np.float = float                # its _gen_timing_signal still uses this alias, removed in numpy 1.24
    sys.path.insert(0, str(BTC_DIR))
    from btc_model import BTC_model, HParams
    from utils.mir_eval_modules import audio_file_to_features, idx2chord
    config = HParams(**yaml.safe_load((BTC_DIR / "run_config.yaml").read_text()))   # its HParams.load needs PyYAML < 6
    model = BTC_model(config=config.model)
    checkpoint = torch.load(BTC_DIR / "test" / "btc_model.pt", map_location="cpu", weights_only=False)
    mean, std = checkpoint["mean"], checkpoint["std"]
    model.load_state_dict(checkpoint["model"])
    model.eval()
    n_timestep = config.model["timestep"]

    def run(audio: Path):
        feature, time_unit, _ = audio_file_to_features(str(audio), config)
        feature = (feature.T - mean) / std
        n_frames = feature.shape[0]
        num_pad = n_timestep - (n_frames % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        predictions = []
        with torch.no_grad():
            for t in range(feature.shape[1] // n_timestep):
                attn, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                prediction, _ = model.output_layer(attn)
                predictions.append(prediction.squeeze().numpy())
        path = np.concatenate(predictions)[:n_frames]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(path)) + 1])
        edges = np.concatenate([starts, [n_frames]]) * time_unit
        return np.stack([edges[:-1], edges[1:]], axis=1), [idx2chord[path[s]] for s in starts]
    return run


def madmom_runner():
    """CNN chord features + CRF decoder, default settings."""
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
    features, decode = CNNChordFeatureProcessor(), CRFChordRecognitionProcessor()

    def run(audio: Path):
        chords = decode(features(str(audio)))
        return np.stack([chords["start"], chords["end"]], axis=1), [str(c) for c in chords["label"]]
    return run


def chordino_runner():
    """Chordino's simplechord output, default parameters, on the file's native rate, mono."""
    os.environ["VAMP_PATH"] = str(REFS / "plugins")
    import soundfile
    import vamp

    def run(audio: Path):
        y, sr = soundfile.read(str(audio), dtype="float32", always_2d=True)
        events = vamp.collect(y.mean(axis=1), sr, "nnls-chroma:chordino", output="simplechord")["list"]
        starts = np.array([e["timestamp"].to_float() for e in events])
        ends = np.concatenate([starts[1:], [len(y) / sr]])
        return np.stack([starts, ends], axis=1), [e["label"] for e in events]
    return run


RUNNERS = {"btc": btc_runner, "madmom": madmom_runner, "chordino": chordino_runner}


# ====================================================== labels ========================================================

SIMPLE = re.compile(r"^([A-G][b#]*)(m(?![a]))?([^/]*)(?:/.*)?$")


def harte(label: str) -> str:
    """Chordino's simple syntax to Harte, keeping only what decides the third.

    "C" -> "C:maj", "Am7" -> "A:min", "F#m7b5" -> "F#:min", "Bbmaj7" -> "Bb:maj", "Cdim7" -> "C:dim",
    "Caug" -> "C:aug", "C6/E" -> "C:maj", and labels already in Harte syntax stay as they are.
    """
    if label == "N" or ":" in label:
        return label
    match = SIMPLE.match(label)
    if match is None:
        raise ValueError(f"unknown label {label!r}")
    root, minor, rest = match.groups()
    if rest.startswith("dim"):
        return f"{root}:dim"
    if rest.startswith("aug"):
        return f"{root}:aug"
    return f"{root}:min" if minor else f"{root}:maj"


def to_majmin(label: str) -> tuple[str, bool]:
    """Raw label -> (one of the 24 V5/V6 classes or N, whether a chord with no third became N)."""
    if label == "N":
        return "N", False
    index = data_v5.target(harte(label))
    return ("N", True) if index < 0 else (LABELS[index], False)


# ====================================================== scoring =======================================================

def lab_path(tool: str, split: str, dataset: str, song_id: str) -> Path:
    return LABS / tool / split / dataset / f"{song_id}.lab"


def write_lab(path: Path, intervals: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{s:.6f}\t{e:.6f}\t{c}\n" for (s, e), c in zip(intervals, labels)))


def read_lab(path: Path) -> tuple[np.ndarray, list[str]]:
    rows = [line.split("\t") for line in path.read_text().splitlines() if line]
    return np.array([[float(s), float(e)] for s, e, _ in rows]), [c for _, _, c in rows]


def score_song(take, intervals: np.ndarray, raw: list[str]) -> dict:
    keep = intervals[:, 1] > intervals[:, 0]
    intervals, raw = intervals[keep], [c for c, k in zip(raw, keep) if k]
    mapped = [to_majmin(c) for c in raw]
    estimate = ChordAnnotation(intervals, [m[0] for m in mapped])
    s = evaluate.score(take.annotation, estimate)
    no_third = float(sum(e - b for (b, e), m in zip(intervals, mapped) if m[1]))
    return {"wcsr": s.wcsr, "total_s": s.total_s, "scored_s": s.scored_s, "matched_s": s.matched_s,
            "reduced_s": s.reduced_s, "excluded_s": s.excluded_s, "n_segments": len(raw), "no_third_s": no_third}


def run_tool(tool: str, split: str, writer, handle) -> None:
    runner = None
    for dataset, takes in run_v5.evaluation_sets(split).items():
        if dataset not in PAIRS[tool]:
            continue
        for take in takes:
            song_id = take.key.split("/", 1)[1]
            row = {"tool": tool, "split": split, "dataset": dataset, "song_id": song_id, "status": "ok"}
            lab, started = lab_path(tool, split, dataset, song_id), time.time()
            try:
                if not lab.exists():
                    runner = runner or RUNNERS[tool]()
                    write_lab(lab, *runner(take.audio))
                    row["seconds"] = round(time.time() - started, 2)
                row.update(score_song(take, *read_lab(lab)))
            except Exception as error:
                row["status"] = f"failed: {type(error).__name__}: {error}"
                print(f"FAILED {tool} {split} {dataset} {song_id}\n{traceback.format_exc()}", flush=True)
            writer.writerow(row)
            handle.flush()
            print(f"{tool} {split} {dataset} {song_id} {row['status']} {row.get('seconds', 'cached')} s", flush=True)


def v6_rows() -> list[dict]:
    """The frozen V6 (bidirectional, seed 1, HMM at p_self 0.4), from its own result files."""
    rows = []
    for split in ("val", "test"):
        summary = json.loads((V6_RUN / split / "summary.json").read_text())
        with open(V6_RUN / split / "per_song.csv") as f:
            songs = [r for r in csv.DictReader(f) if r["system"] == str(V6_P_SELF)]
        for dataset in ("rwc", "guitarset"):
            pooled = next(r["wcsr"] for r in summary["rows"]
                          if r["dataset"] == dataset and r["system"] == V6_P_SELF)
            per_song = [float(r["wcsr"]) for r in songs if r["dataset"] == dataset]
            rows.append({"split": split, "dataset": dataset, "pooled_wcsr": pooled,
                         "mean_song_wcsr": float(np.mean(per_song)), "n_songs": len(per_song)})
    return rows


def summarise() -> None:
    with open(PER_SONG) as f:
        rows = {(r["tool"], r["split"], r["dataset"], r["song_id"]): r for r in csv.DictReader(f)}   # last run wins
    v6 = {(r["split"], r["dataset"]): r for r in v6_rows()}
    table, failed = [], [r for r in rows.values() if r["status"] != "ok"]
    for tool, split, dataset in sorted({k[:3] for k in rows}):
        songs = [r for k, r in rows.items() if k[:3] == (tool, split, dataset) and r["status"] == "ok"]
        scores = [evaluate.SongScore(float(r["total_s"]), float(r["scored_s"]), float(r["matched_s"]),
                                     float(r["reduced_s"])) for r in songs]
        total = evaluate.pooled(scores)
        reference = v6[(split, dataset)]
        table.append({"tool": tool, "split": split, "dataset": dataset, "n_songs": len(songs),
                      "pooled_wcsr": total.wcsr, "mean_song_wcsr": float(np.mean([s.wcsr for s in scores])),
                      "scored_s": total.scored_s, "excluded_s": total.excluded_s, "reduced_s": total.reduced_s,
                      "no_third_s": sum(float(r["no_third_s"]) for r in songs),
                      "runtime_s": sum(float(r["seconds"] or 0) for r in songs),
                      "v6_pooled_wcsr": reference["pooled_wcsr"], "v6_mean_song_wcsr": reference["mean_song_wcsr"]})
    (RESULTS / "summary.json").write_text(json.dumps(
        {"metric": "majmin, evaluate.score; pooled_wcsr = evaluate.pooled, as in reports/v6.md",
         "v6": f"{V6_RUN.relative_to(REPO)}, p_self {V6_P_SELF}", "rows": table,
         "failed": [{k: r[k] for k in ("tool", "split", "dataset", "song_id", "status")} for r in failed]}, indent=2))
    print(f"\n{'tool':<9}{'split':<6}{'dataset':<11}{'songs':>6}{'WCSR':>8}{'V6':>8}{'seconds':>9}")
    for r in table:
        print(f"{r['tool']:<9}{r['split']:<6}{r['dataset']:<11}{r['n_songs']:>6}{100 * r['pooled_wcsr']:>8.2f}"
              f"{100 * r['v6_pooled_wcsr']:>8.2f}{r['runtime_s']:>9.0f}")
    print(f"failed: {len(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tools", nargs="+", choices=sorted(RUNNERS), default=sorted(RUNNERS))
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val"])
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if not args.summary_only:
        new = not PER_SONG.exists()
        with open(PER_SONG, "a", newline="") as handle:
            writer = csv.DictWriter(handle, FIELDS)
            if new:
                writer.writeheader()
            for split in args.splits:
                for tool in args.tools:
                    run_tool(tool, split, writer, handle)
    summarise()


if __name__ == "__main__":
    main()
