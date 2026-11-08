"""V11's unlabelled pool: list the recordings, drop leakage and duplicates, cache V8's causal log-CQT.

    python scripts/v11_pool.py manifest        # data/v11_pool/manifest.csv and excluded.csv
    python scripts/v11_pool.py cache --jobs 8  # data/processed/v11_pool_Fs1/<id>.npy (float16), resumable

The pool is data/unlabelled_audio/ (git-ignored, read in place). Only files libsndfile reads are listed
(FLAC, WAV, AIFF); ALAC .m4a is left out. A recording is excluded when:
  - a folder on its path is, or starts with, an artist of a ChoCo val or test recording ("Sting",
    "Sting - ..."), since ChoCo's uspop2002 part is split by artist; RWC-P and GuitarSet share no artist;
  - its title (file name without track number or extension) equals the title of a ChoCo val or test
    recording, or starts with it as whole words ('fields of gold s'): other takes, remasters, covers; RWC-P's songs were written for the database and
    GuitarSet's are its own, so neither has a title to match;
  - it is shorter than 10 s;
  - it repeats an earlier file: same normalised title and the same duration to the second.
The manifest and the exclusion list stay in data/ because they list a private music library.
Features: features_v7_1.causal_log_cqt on configs/v11.yaml's grid (C2, 24 bins per octave, live-path
tuning estimate), the same features the play-along path computes.
"""
import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from chordtracker import data_choco, features_v7_1, paths

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402

POOL = paths.REPO_ROOT / "data" / "unlabelled_audio"
OUT = paths.REPO_ROOT / "data" / "v11_pool"
CACHE = paths.REPO_ROOT / "data" / "processed" / "v11_pool_Fs1"
AUDIO = {".flac", ".wav", ".aif", ".aiff"}
MIN_SECONDS = 10.0


def config() -> dict:
    return run_v5.load_config(paths.REPO_ROOT / "configs" / "v11.yaml")


def norm(text: str) -> str:
    """Lower case, '&' as 'and', bracketed parts and punctuation dropped: 'Fields Of Gold (Live)' -> 'fields of gold'."""
    text = re.sub(r"[\(\[].*?[\)\]]", " ", text.lower().replace("&", " and "))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def title_of(path: Path) -> str:
    """'03 Fields Of Gold.flac' -> 'fields of gold'; '1-03 - Song.wav' -> 'song'."""
    return norm(re.sub(r"^[\d\s\-_.]+(?=\D)", "", path.stem))


def held_out() -> tuple[set[str], set[str]]:
    """Normalised artists and titles of every ChoCo val and test recording (manifest metadata only)."""
    cfg = config()
    rows = data_choco.manifest(cfg)
    split = data_choco.load_split(data_choco.CHOCO_SPLIT)
    ids = split["val"] + split["test"]
    artists = {norm(rows[i]["artist"]) for i in ids}
    titles = {norm(rows[i]["title"]) for i in ids}
    return artists, titles


def artist_match(path: Path, artists: set[str]) -> str | None:
    for part in path.relative_to(POOL).parts[:-1]:
        for a in artists:
            if norm(part) == a or norm(part.split(" - ")[0]) == a:
                return a
    return None


def build_manifest() -> None:
    artists, titles = held_out()
    kept, excluded, seen = [], [], set()
    for path in sorted(p for p in POOL.rglob("*") if p.suffix.lower() in AUDIO and not p.name.startswith("._")):
        rel = str(path.relative_to(POOL))
        try:
            seconds = sf.info(str(path)).duration
        except RuntimeError as e:
            excluded.append({"path": rel, "reason": f"unreadable: {e}"})
            continue
        title = title_of(path)
        artist = artist_match(path, artists)
        if artist:
            reason = f"val/test artist {artist}"
        elif any(title == t or title.startswith(t + " ") for t in titles):
            reason = f"val/test title {title}"
        elif seconds < MIN_SECONDS:
            reason = f"shorter than {MIN_SECONDS:.0f} s"
        elif (title, round(seconds)) in seen:
            reason = "duplicate"
        else:
            seen.add((title, round(seconds)))
            kept.append({"id": f"pool_{len(kept):04d}", "path": rel, "seconds": f"{seconds:.1f}"})
            continue
        excluded.append({"path": rel, "reason": reason})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (("manifest.csv", kept, ["id", "path", "seconds"]),
                               ("excluded.csv", excluded, ["path", "reason"])):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    hours = sum(float(r["seconds"]) for r in kept) / 3600
    print(f"kept {len(kept)} recordings, {hours:.1f} h; excluded {len(excluded)}")
    for r in excluded:
        if not r["reason"].startswith("duplicate"):
            print(f"  excluded {r['path']}: {r['reason']}")


def pool_rows() -> list[dict]:
    with open(OUT / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def cache_one(row: dict, cfg: dict) -> str:
    file = CACHE / f"{row['id']}.npy"
    if not file.exists():
        cqt = features_v7_1.causal_log_cqt(POOL / row["path"], cfg)
        tmp = file.with_suffix(".tmp.npy")
        np.save(tmp, cqt.astype(np.float16))
        tmp.rename(file)
    return row["id"]


def build_cache(jobs: int) -> None:
    cfg = config()
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = [r for r in pool_rows() if not (CACHE / f"{r['id']}.npy").exists()]
    print(f"{len(rows)} recordings to cache", flush=True)
    with ProcessPoolExecutor(jobs) as ex:
        futures = [ex.submit(cache_one, r, cfg) for r in rows]
        for done, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if done % 50 == 0 or done == len(rows):
                print(f"  {done} / {len(rows)}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["manifest", "cache"])
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if args.step == "manifest":
        build_manifest()
    else:
        build_cache(args.jobs)


if __name__ == "__main__":
    main()
