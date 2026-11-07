"""Score V8's frozen points on test, once: RWC-P, GuitarSet and the ChoCo export's test split.

    python scripts/v8_test.py --freeze      # -> results/v8/test/summary.json

configs/v8.yaml `frozen.points` holds the choices made on validation (vocabulary, seed, decoder,
p_self, lag). Each point is decoded with the config's prior power and transition matrix, scored with
mir_eval majmin and sevenths pooled per set, and only the summary is written: no per-song rows and no
confusion matrix, so test is evaluated, never inspected. The script refuses to run a second time.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import data_choco, paths, train_v7_2, transitions_v8

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config, evaluation_sets)
import run_v8  # noqa: E402  (load_song, model_for, decoder_parts, score_song, summarize)

OUT = paths.REPO_ROOT / "results" / "v8" / "test"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v8.yaml")
    parser.add_argument("--freeze", action="store_true", required=True, help="the val choices are final")
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args(argv)


def test_takes(cfg: dict) -> dict[str, list]:
    return {**run_v5.evaluation_sets("test"), "choco": data_choco.choco_takes("test", cfg, frozen=True)}


def point_rows(point: dict, cfg: dict, takes: dict[str, list]) -> list[dict]:
    """Pooled rows (per set and mean) of one frozen point on the given takes."""
    run = paths.REPO_ROOT / "results" / "v8" / f"{point['vocab']}_seed{point['seed']}"
    model = run_v8.model_for(run, run_v8.n_classes(point["vocab"]), cfg)
    prior = np.array(json.loads((run / "prior.json").read_text()))
    Q = run_v8.decoder_parts(cfg, point["vocab"], prior)
    decode_prior = transitions_v8.tempered_prior(prior, cfg["decoder"].get("prior_power", 1.0))
    per_song = []
    for dataset, ts in takes.items():
        for take in ts:
            song = run_v8.load_song(take, cfg, point["vocab"])
            post = train_v7_2.song_posteriors(model, song, cfg["train"]["input_lookahead"])
            rows, _ = run_v8.score_song(post, song, take, cfg, point["vocab"], Q, decode_prior, dataset, dataset)
            per_song += [r for r in rows
                         if (r["system"], r["p_self"], r["lag"]) == (point["system"], point["p_self"], point["lag"])]
    return run_v8.summarize(per_song)


def main(argv=None) -> None:
    args = parse_args(argv)
    if (args.out / "summary.json").exists():
        raise SystemExit(f"{args.out / 'summary.json'} exists: test is scored once")
    cfg = run_v5.load_config(args.config)
    if cfg["transitions"].get("source", "billboard") != "billboard":
        raise SystemExit("the frozen points use the Billboard transition matrix")
    takes = test_takes(cfg)
    points = []
    for point in cfg["frozen"]["points"]:
        rows = point_rows(point, cfg, takes)
        result = {**point,
                  "test": {r["dataset"]: {m: r[m] for m in run_v8.METRICS} for r in rows},
                  "test_changes_per_min": {r["dataset"]: r["mean_changes_per_min"] for r in rows
                                           if "mean_changes_per_min" in r},
                  "n_takes": {r["dataset"]: r["n_takes"] for r in rows if "n_takes" in r}}
        points.append(result)
        print(json.dumps({k: result[k] for k in ("name", "test")}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(
        {"version": "v8", "split": "test", "prior_power": cfg["decoder"].get("prior_power", 1.0),
         "points": points}, indent=2))


if __name__ == "__main__":
    main()
