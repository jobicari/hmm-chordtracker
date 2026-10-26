"""Fetch AAM mixes one song at a time and keep only their log-CQT cache.

AAM (Artificial Audio Multitracks, Zenodo record 5794629, v1.1.0, CC BY 4.0) ships its 1,000
mixes in one 14.71 GB zip. Each FLAC is read out of that zip on Zenodo with HTTP range requests,
checked against the CRC-32 the zip stores for it, turned into a log-CQT cache entry with V5's
settings, and deleted. Only the cache stays on disk. The annotations zip (10 MB) is downloaded
whole, checked against its md5 and unpacked to data/aam/annotations/.

The run can be stopped and restarted: songs listed in data/aam/fetched.txt are skipped.

    python scripts/fetch_aam.py --dry-run      # 2 songs, then stop
    python scripts/fetch_aam.py --shard 2/6    # every 6th song (song % 6 == 1); run 1/6 .. 6/6 side by side
    python scripts/fetch_aam.py                # all 1,000
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import shutil
import socket
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from chordtracker import paths
from chordtracker.features_v5 import cached_log_cqt

sys.path.insert(0, str(paths.REPO_ROOT / "scripts"))
from run_v5 import load_config  # noqa: E402

_getaddrinfo = socket.getaddrinfo


def _ipv4_only(host, port, family=0, *args, **kwargs):
    """Resolve to IPv4 addresses only: Zenodo's IPv6 route hangs, and each request would wait for it."""
    return _getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)


socket.getaddrinfo = _ipv4_only

RECORD = "https://zenodo.org/records/5794629/files"
MIXES_ZIP = f"{RECORD}/0001-1000-audio-mixes.zip"
ANNOTATIONS_ZIP = f"{RECORD}/0001-1000-annotations-v1.1.0.zip"
ANNOTATIONS_MD5 = "fab44c247f53292e7f4c37d4850aed70"
AAM_DIR = paths.REPO_ROOT / "data" / "aam"
RETRIES = 5


def _get(url: str, headers: dict | None = None, method: str = "GET"):
    """urlopen with retries and a growing wait (2, 4, 8, 16 s)."""
    for attempt in range(RETRIES):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=headers or {}, method=method), timeout=60
            )
        except OSError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** (attempt + 1))


class RemoteFile(io.RawIOBase):
    """A seekable, read-only view of a file on a server that answers HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.size = int(_get(url, method="HEAD").headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        for attempt in range(RETRIES):      # a dropped or timed-out read is retried, not only the request
            try:
                data = _get(self.url, {"Range": f"bytes={self.pos}-{end}"}).read()
                break
            except (OSError, http.client.HTTPException):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_annotations() -> None:
    """Download the annotations zip, check its md5 and unpack it."""
    target = AAM_DIR / "annotations"
    if target.exists():
        return
    data = _get(ANNOTATIONS_ZIP).read()
    md5 = hashlib.md5(data).hexdigest()
    if md5 != ANNOTATIONS_MD5:
        raise SystemExit(f"annotations md5 {md5} != {ANNOTATIONS_MD5}")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(target)
    print(f"annotations: {len(data) / 1e6:.1f} MB, md5 ok -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=paths.REPO_ROOT / "configs" / "v5.yaml")
    parser.add_argument("--dry-run", action="store_true", help="2 songs, then stop")
    parser.add_argument("--shard", default="1/1", help="i/n: only songs with song %% n == i - 1")
    args = parser.parse_args()
    cfg = load_config(args.config)

    AAM_DIR.mkdir(parents=True, exist_ok=True)
    fetch_annotations()

    done_file = AAM_DIR / "fetched.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    archive = zipfile.ZipFile(io.BufferedReader(RemoteFile(MIXES_ZIP), 8 << 20))
    songs = sorted((i for i in archive.infolist() if i.filename.endswith("_mix.flac")),
                   key=lambda i: i.filename)
    todo = [i for i in songs if Path(i.filename).name.split("_")[0] not in done]
    shard, n_shards = map(int, args.shard.split("/"))
    todo = [i for i in todo if int(Path(i.filename).name.split("_")[0]) % n_shards == shard - 1]
    if args.dry_run:
        todo = todo[:2]
    print(f"{len(songs)} mixes in the zip, {len(done)} already cached, {len(todo)} to do")

    start = time.time()
    for n, info in enumerate(todo, 1):
        song = Path(info.filename).name.split("_")[0]
        with tempfile.TemporaryDirectory() as tmp:
            flac = Path(tmp) / f"{song}_mix.flac"
            # ZipExtFile checks the stored CRC-32 at the end of the read and raises on a mismatch.
            with archive.open(info) as src, flac.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            cached_log_cqt(flac, f"aam/{song}", cfg)
        with done_file.open("a") as f:
            f.write(song + "\n")
        rate = (time.time() - start) / n
        print(f"[{args.shard}] {song}  {info.file_size / 1e6:5.1f} MB  {n}/{len(todo)}  "
              f"~{rate * (len(todo) - n) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
