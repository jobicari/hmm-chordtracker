"""Downbeats (and beats) estimated by madmom on the RWC-P and GuitarSet validation audio, offline.

    ~/miniconda3/envs/chord-refs/bin/python scripts/v13_madmom_downbeats.py

Runs in the separate chord-refs environment (madmom 0.17), not the project's: it imports madmom and the
standard library only. Reads results/v13/est_downbeats/audio.tsv (key, audio path; written by
scripts/v12_beats.py --list-audio) and writes one <key with / as __>.txt per song: time in seconds and
position in the bar (1 = downbeat), as madmom's RNNDownBeatProcessor + DBNDownBeatTrackingProcessor
(3 or 4 beats a bar, 100 frames a second) give them. Both read the whole recording: an offline estimate.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "v13" / "est_downbeats"


def track(row: tuple[str, str]) -> str:
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
    key, audio = row
    out = OUT / f"{key.replace('/', '__')}.txt"
    if not out.exists():
        act = RNNDownBeatProcessor()(audio)
        beats = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(act)
        out.write_text("".join(f"{t:.3f}\t{int(p)}\n" for t, p in beats))
    return key


def main() -> None:
    rows = [tuple(line.split("\t")) for line in (OUT / "audio.tsv").read_text().splitlines() if line]
    with Pool(8) as pool:
        for i, key in enumerate(pool.imap_unordered(track, rows), 1):
            print(f"{i}/{len(rows)} {key}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
