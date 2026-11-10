"""Component 37 (V14): the input level of every frame, in dBFS of the raw mono audio, from past audio only.

The log-CQT is relative to the song's running level (features_v7_1.running_level), so near-silence and a
quiet chord can look alike to the network. V14's level variant gives the GRU one more number per frame:

    level(t) = 10 log10(mean of y[n] ** 2 over the WINDOW samples t * hop - WINDOW <= n < t * hop)

dBFS of the raw mono input (a full-scale square wave is 0 dB), on the CQT's frame grid (frame t is centred
at sample t * hop) and from the samples before frame t's centre only, so it is known before the CQT
column of frame t (which reads audio up to frame t + 1). Samples before the song are silence. The value
is clipped at FLOOR_DB (-100 dB; digital silence is -inf) and divided by -FLOOR_DB, so the network reads
about -1 (silence) to 0 (full scale).

    window_level(samples)              one window -> its scaled level
    frame_levels(y, hop, n_frames)     (n_frames,) scaled level of every frame of a whole recording
    StreamingLevels(hop)               the same values block by block: step(block) -> new frames' levels
    cache_dir() / cached_levels(take, cfg, n_frames)   data/processed/v14_dbfs/<key>.npy

Offline (cache) and live use the same window_level on the same samples, so they agree exactly; a test
checks it. The level cache sits next to the log-CQT caches; the CQT caches are not touched.
"""
from pathlib import Path

import librosa
import numpy as np

from chordtracker import paths

FLOOR_DB = -100.0
WINDOW = 2048                    # one frame of samples at hop 2,048: the hop before the frame's centre


def window_level(samples: np.ndarray) -> float:
    """dBFS of one window (float64 mean square), clipped at FLOOR_DB, scaled to [-1, 0] (above 0 dB stays > 0).

    Example: a full-scale square wave -> 0.0, a sine of amplitude 1 -> -0.0301, silence -> -1.0.
    """
    power = float(np.mean(np.square(samples, dtype=np.float64)))
    db = 10.0 * np.log10(power) if power > 0 else FLOOR_DB
    return max(db, FLOOR_DB) / -FLOOR_DB


def _window(y: np.ndarray, t: int, hop: int, offset: int = 0) -> np.ndarray:
    """The WINDOW samples before frame t's centre, zeros outside y; y[0] is sample `offset` of the recording."""
    lo, hi = t * hop - WINDOW - offset, t * hop - offset
    out = np.zeros(WINDOW, dtype=np.float32)
    a, b = max(lo, 0), min(hi, len(y))
    if a < b:
        out[a - lo:b - lo] = y[a:b]
    return out


def frame_levels(y: np.ndarray, hop: int, n_frames: int | None = None) -> np.ndarray:
    """(n_frames,) float32 scaled level of frames 0 .. n_frames - 1; n_frames defaults to librosa's 1 + len // hop."""
    n = 1 + len(y) // hop if n_frames is None else n_frames
    y = np.asarray(y, dtype=np.float32)
    return np.array([window_level(_window(y, t, hop)) for t in range(n)], dtype=np.float32)


class StreamingLevels:
    """frame_levels block by block: step(block) returns the level of every frame whose window is complete.

    Frame t's window ends at sample t * hop, so it is complete once t * hop samples have arrived (frame 0 at
    once: its window is all before the song). Only the samples a later window can read are kept. The caller
    pairs level t with log-CQT column t.
    """

    def __init__(self, hop: int):
        self.hop = hop
        self.raw = np.zeros(0, dtype=np.float32)
        self.offset = 0                                       # recording index of raw[0]
        self.frames = 0

    def step(self, block: np.ndarray) -> list[np.float32]:
        self.raw = np.concatenate([self.raw, np.asarray(block, dtype=np.float32)])
        out = []
        while self.frames * self.hop <= self.offset + len(self.raw):
            out.append(np.float32(window_level(_window(self.raw, self.frames, self.hop, self.offset))))
            self.frames += 1
        drop = max(0, self.frames * self.hop - WINDOW - self.offset)    # before the next frame's window
        self.raw, self.offset = self.raw[drop:], self.offset + drop
        return out


def cache_dir() -> Path:
    return paths.DATA_DIR / "processed" / "v14_dbfs"


def cached_levels(audio: Path, key: str, cfg: dict, n_frames: int) -> np.ndarray:
    """frame_levels of the audio (cfg's sr and mono, as the CQT reads it), saved as <key>.npy the first time."""
    file = cache_dir() / f"{key}.npy"
    if file.exists():
        out = np.load(file)
    else:
        y, _ = librosa.load(str(audio), sr=cfg["audio"]["sr"], mono=cfg["audio"]["mono"])
        out = frame_levels(y, cfg["cqt"]["hop_length"])
        file.parent.mkdir(parents=True, exist_ok=True)
        np.save(file, out)
    if len(out) != n_frames:
        raise ValueError(f"{key}: {len(out)} level frames, {n_frames} log-CQT columns")
    return out
