"""How much temporal context the CRNN uses: val frame accuracy when the GRU may only read chunks of a song.

    python scripts/v6_context_check.py results/v6/bidir_seed1 results/v6/causal_seed1   # -> <run>/context_check.json

Each val song is read in back-to-back chunks of `length` frames (data_v6.excerpt, so every frame
keeps its own 15-frame window), and the GRU starts from zero at every chunk. length 1 leaves the
GRU nothing but the frame itself: the front end alone, as close to V5's frame classifier as this
model gets. The whole song is how run_v6 reads it. If accuracy rises with the chunk length, the GRU
uses the frames around its window. Val only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from chordtracker import data_v5, data_v6, model_v6, paths, train_v6

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402

LENGTHS = [1, 4, 16, 64, 128, 512, None]          # None: the whole song as one chunk


def chunked_posteriors(model, song: data_v5.Song, length: int | None) -> np.ndarray:
    """(24, n_frames) posteriors with the GRU restarted every `length` frames."""
    n = song.cqt.shape[1]
    length = length or n
    parts = [model_v6.frame_posteriors(model, torch.from_numpy(data_v6.excerpt(song, s, 0, min(length, n - s)))[None, None])
             for s in range(0, n, length)]
    return torch.cat(parts, dim=1).numpy()


def main() -> None:
    cfg = run_v5.load_config(paths.REPO_ROOT / "configs" / "v6.yaml")
    val = {name: [data_v5.load(take, cfg) for take in takes] for name, takes in run_v5.evaluation_sets("val").items()}
    for run in map(Path, sys.argv[1:]):
        bidirectional = json.loads((run / "val" / "summary.json").read_text())["config"]["train"]["bidirectional"]
        model = train_v6.load(run / "model.pt", bidirectional)
        rows = []
        for length in LENGTHS:
            row = {"chunk_frames": length or "song"}
            for name, songs in val.items():
                right = total = 0
                for song in songs:
                    keep = song.targets >= 0
                    right += int((chunked_posteriors(model, song, length).argmax(axis=0)[keep] == song.targets[keep]).sum())
                    total += int(keep.sum())
                row[name] = right / total
            row["mean"] = float(np.mean([row[name] for name in val]))
            rows.append(row)
            print(f"{run.name}  chunk {str(row['chunk_frames']):>5}  " + "  ".join(
                f"{k} {100 * row[k]:.2f}%" for k in [*val, "mean"]), flush=True)
        (run / "context_check.json").write_text(json.dumps({"split": "val", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
