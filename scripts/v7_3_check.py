"""V7.3 end-to-end check: do the streaming (direct) features give the same chords as the V7.1/V7.2 cache?

    python scripts/v7_3_check.py           # -> results/v7_3/val_check.json

For the frozen V7.2 points at 0.28 s, 0.46 s and 0.65 s, both seeds of each point's network decode every
validation song twice: from the cached features (librosa's CQT) and from streaming_features.offline_log_cqt,
the whole-file run of exactly the computation StreamingFeatures does block by block (the tests check they
give the same columns). Aligned WCSR, mean of RWC-P and GuitarSet. Validation only.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

from chordtracker import acoustic_hmm, data_v5, data_v7_2, paths, streaming_features, train_v7_2

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402
import run_v7  # noqa: E402
import run_v7_2  # noqa: E402

POINTS = [  # V7.2's frozen points (configs/v7_2.yaml)
    dict(total=3, cqt_from="C2", input_lookahead=2, system=run_v7.ARGMAX, lag=0, p_self=None),
    dict(total=5, cqt_from="F#0", input_lookahead=2, system=run_v7.FIXED_LAG, lag=2, p_self=0.7),
    dict(total=7, cqt_from="F#0", input_lookahead=2, system=run_v7.FIXED_LAG, lag=4, p_self=0.98),
]


def direct_song(take, cfg) -> data_v5.Song:
    """The take's direct (streaming) log-CQT, cached under data/processed/v7_3_direct_<fmin>/."""
    file = paths.DATA_DIR / "processed" / f"v7_3_direct_{cfg['cqt']['fmin'].replace('#', 's')}" / f"{take.key}.npy"
    if file.exists():
        cqt = np.load(file).astype(np.float32)
    else:
        y, _ = librosa.load(str(take.audio), sr=cfg["audio"]["sr"], mono=True)
        cqt = streaming_features.offline_log_cqt(y, cfg)
        file.parent.mkdir(parents=True, exist_ok=True)
        np.save(file, cqt.astype(np.float16))
    times = acoustic_hmm.frame_times(cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
    return data_v5.Song(take.key, cqt, data_v5.frame_targets(take.annotation, times))


def main() -> None:
    base = run_v5.load_config(paths.REPO_ROOT / "configs" / "v7_2.yaml")
    out = []
    for point in POINTS:
        for seed in (0, 1):
            cfg = run_v7_2.cell_config(base, point["cqt_from"], point["input_lookahead"], seed)
            run = run_v7_2.run_dir(point["cqt_from"], point["input_lookahead"], seed)
            model = train_v7_2.load(run / "model.pt", False, 0.5)
            prior = np.array(json.loads((run / "prior.json").read_text()))
            Q = run_v5.change_prior(cfg)
            grids = {run_v7.FILTER: [], run_v7.OFFLINE: [],
                     run_v7.FIXED_LAG: {point["p_self"]: [point["lag"]]} if point["system"] == run_v7.FIXED_LAG else {}}
            row = {**point, "seed": seed}
            for source, load in (("cache", data_v7_2.load), ("streaming", direct_song)):
                scores = []
                for dataset, takes in run_v5.evaluation_sets("val").items():
                    matched = scored = 0.0
                    diffs = []
                    for take in takes:
                        song = load(take, cfg)
                        post = train_v7_2.song_posteriors(model, song, point["input_lookahead"])
                        times = acoustic_hmm.frame_times(post.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
                        for system, p_self, lag, path in run_v7.frame_paths(post, Q, prior, grids):
                            if (system, lag, p_self) == (point["system"], point["lag"], point["p_self"]):
                                s, _ = run_v7.score(path, times, take)
                                matched += s.matched_s
                                scored += s.scored_s
                    scores.append(100 * matched / scored)
                row[source] = float(np.mean(scores))
            row["difference"] = row["streaming"] - row["cache"]
            print(row, flush=True)
            out.append(row)
    file = paths.REPO_ROOT / "results" / "v7_3" / "val_check.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
