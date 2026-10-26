"""V5.2 step 0: are the cached AAM songs and their beat labels fit to train on?

    python scripts/v5_2_check_aam.py                 # all cached AAM songs, template check on 50
    python scripts/v5_2_check_aam.py --songs 10      # quicker

Reports, written to results/v5.2/aam_check.json:
- totals: songs, hours, frames with a target, beats, and the label shares (major, minor, N, X);
- chord balance: the share of each of the 24 classes, against RWC-P train's;
- tail: seconds of audio after the last beat (no target there);
- template check: a chord template on the log-CQT folded to 12 pitch classes, frame accuracy against
  the labels, on N AAM songs and on the 60 RWC-P train songs with the same code. AAM's labels are
  exact, so its score should be at least RWC-P's; a much lower one would mean misaligned labels.
  Run with the labels moved by -3..+3 frames (+-0.28 s): the best shift should be 0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chordtracker import acoustic_hmm, data_aam, data_v5, paths

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402  (load_config)


SHIFTS = range(-3, 4)


def chroma(cqt: np.ndarray) -> np.ndarray:
    """(168, n) log-CQT, 2 bins per semitone from F#0 -> (12, n), C = 0, each frame unit length."""
    pitch_class = (6 + np.arange(cqt.shape[0]) // 2) % 12
    folded = np.stack([cqt[pitch_class == p].sum(axis=0) for p in range(12)])
    return folded / (np.linalg.norm(folded, axis=0, keepdims=True) + 1e-9)


def templates() -> np.ndarray:
    """(24, 12) unit triads: C:maj..B:maj (root, +4, +7), then C:min..B:min (root, +3, +7)."""
    rows = []
    for third in (4, 3):
        for root in range(12):
            t = np.zeros(12)
            t[[root, (root + third) % 12, (root + 7) % 12]] = 1
            rows.append(t / np.linalg.norm(t))
    return np.array(rows)


def template_accuracy(songs: list[data_v5.Song]) -> dict[int, float]:
    """Pooled frame accuracy of the best-matching template, with the targets moved by each shift."""
    T = templates()
    right, total = dict.fromkeys(SHIFTS, 0), dict.fromkeys(SHIFTS, 0)
    for song in songs:
        guess = (T @ chroma(song.cqt)).argmax(axis=0)
        for s in SHIFTS:
            y = np.roll(song.targets, s)
            keep = y >= 0
            if s > 0:
                keep[:s] = False
            elif s < 0:
                keep[s:] = False
            right[s] += int((guess[keep] == y[keep]).sum())
            total[s] += int(keep.sum())
    return {s: right[s] / total[s] for s in SHIFTS}


def class_shares(songs: list[data_v5.Song]) -> np.ndarray:
    """Share of each of the 24 classes among the frames with a target."""
    y = np.concatenate([s.targets for s in songs])
    return np.bincount(y[y >= 0], minlength=24) / (y >= 0).sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v5_2.yaml")
    parser.add_argument("--songs", type=int, default=50, help="AAM songs in the template check")
    args = parser.parse_args()
    cfg = run_v5.load_config(args.config)
    fps = cfg["audio"]["sr"] / cfg["cqt"]["hop_length"]

    ids = data_aam.aam_song_ids()
    takes = data_aam.aam_takes(ids)
    songs = [data_v5.load(t, cfg) for t in takes]
    labels = [label for t in takes for label in t.annotation.labels]
    kinds = {"major": sum(l.endswith(":maj") for l in labels), "minor": sum(l.endswith(":min") for l in labels),
             "N": labels.count("N"), "X": labels.count("X")}
    seconds = [s.cqt.shape[1] / fps for s in songs]
    tails = [sec - t.annotation.intervals[-1, 1] for sec, t in zip(seconds, takes)]
    rwc = [data_v5.load(t, cfg) for t in data_v5.rwc_takes("train")]

    report = {
        "songs": len(songs), "hours": sum(seconds) / 3600,
        "song_seconds": [min(seconds), float(np.median(seconds)), max(seconds)],
        "frames_with_target": int(sum((s.targets >= 0).sum() for s in songs)),
        "beats": len(labels), "beat_shares": {k: v / len(labels) for k, v in kinds.items()},
        "tail_seconds": [min(tails), float(np.median(tails)), max(tails)],
        "class_shares_aam": class_shares(songs).tolist(), "class_shares_rwc_train": class_shares(rwc).tolist(),
        "prior_aam": acoustic_hmm.training_prior(np.concatenate([s.targets for s in songs])).tolist(),
        "template_accuracy_aam": template_accuracy(songs[:args.songs]),
        "template_accuracy_rwc_train": template_accuracy(rwc),
    }
    out = paths.REPO_ROOT / "results" / "v5.2" / "aam_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"{report['songs']} songs, {report['hours']:.1f} h, {report['frames_with_target']} frames with a target")
    print("beats: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in report["beat_shares"].items()))
    print(f"tail after the last beat: min {min(tails):.1f} s, median {np.median(tails):.1f} s")
    major = sum(report["class_shares_aam"][:12])
    print(f"major share of frames: AAM {100 * major:.1f}%, RWC-P train {100 * sum(report['class_shares_rwc_train'][:12]):.1f}%")
    for name in ("aam", "rwc_train"):
        acc = report[f"template_accuracy_{name}"]
        print(f"template accuracy {name:>9}: " + "  ".join(f"{s:+d}: {100 * a:.1f}%" for s, a in acc.items()))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
