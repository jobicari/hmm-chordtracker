"""Component 26 (V7.3): V7.1's causal log-CQT computed block by block, as a live system receives audio.

features_v7_1 computes the causal features on a whole file. StreamingFeatures takes the recording one
block of 2,048 samples (one frame, 93 ms) at a time and returns each log-CQT column as soon as it can
be computed, with only what the past requires:

    harmonic part   STFT frames (hop 512, window 2,048, centred, zeros before the song), the time median
                    over the past `harmonic_median_frames` frames and the frequency median of each frame,
                    the soft mask, then overlap-add. Harmonic sample n is final once the audio up to
                    n + 2,048 has arrived (librosa's stft / istft, computed frame by frame).
    CQT             each bin computed directly from its kernel (librosa.filters.wavelet, exact, full rate)
                    on the harmonic audio, centred at frame t - d_k (d_k = the bin's delay, features_v7_1)
                    and scaled by sqrt(kernel length) as librosa does. librosa's CQT downsamples the low
                    octaves, so the two differ slightly there (reports/v7_3.md measures how much).
    level + log     the 99th percentile of every value so far (kept sorted), log(1 + 10 C / level)
    tuning          A440 until tuning_seconds of audio have arrived, then the estimate on them, from the
                    same column as features_v7_1.start_tuning; its harmonic medians are computed frame by
                    frame while those seconds arrive (TuningMedians), so the block that completes them
                    does not redo the whole separation

    step(block)     -> list of new (n_bins,) float32 columns, in frame order (usually one per block)
    flush()         -> the remaining columns at the end of the song (zeros after it, as librosa pads)

offline_log_cqt(y, cfg) runs the same direct computation on a whole file: the streaming code must give
exactly its columns, and both are compared with the V7.1 cache in the tests and the report.
"""
import librosa
import numpy as np
import scipy.fft
from scipy.ndimage import median_filter

from chordtracker import features_v7_1
from chordtracker.features_v7_1 import MEDIAN_FRAMES, N_FFT, STFT_HOP

HALF = N_FFT // 2


class Kernels:
    """The CQT bins of one tuning: trimmed complex kernels, their lengths, and the delay of each bin."""

    def __init__(self, cfg: dict, tuning: float, delays: np.ndarray):
        q = cfg["cqt"]
        freqs = librosa.cqt_frequencies(q["n_bins"], fmin=librosa.note_to_hz(q["fmin"]),
                                        bins_per_octave=q["bins_per_octave"], tuning=tuning)
        filters, lengths = librosa.filters.wavelet(freqs=freqs, sr=cfg["audio"]["sr"], filter_scale=1, norm=1,
                                                   pad_fft=False)
        width = filters.shape[1]
        self.kernels, self.starts = [], []
        for row in filters:
            nz = np.nonzero(row)[0]
            self.kernels.append(row[nz[0]:nz[-1] + 1].astype(np.complex128))
            self.starts.append(nz[0] - width // 2)              # first tap, relative to the centre sample
        self.scale = np.sqrt(lengths)
        self.delays = delays
        self.hop = q["hop_length"]
        self.reach = max(-d * self.hop + s + len(k) for k, s, d in zip(self.kernels, self.starts, delays))

    def needs(self, t: int) -> int:
        """The last harmonic sample (exclusive) column t reads."""
        return t * self.hop + self.reach

    def all_columns(self, y_h: np.ndarray, n: int) -> np.ndarray:
        """(n_bins, n) |CQT| of columns 0 .. n - 1 at once: the same values as column(), vectorized per bin."""
        out = np.empty((len(self.kernels), n))
        for i, (k, s, d) in enumerate(zip(self.kernels, self.starts, self.delays)):
            lo = (np.arange(n) - d) * self.hop + s                      # first sample of every column
            pad_lo = max(0, -int(lo.min()))
            pad_hi = max(0, int(lo.max()) + len(k) - len(y_h))
            padded = np.concatenate([np.zeros(pad_lo, y_h.dtype), y_h, np.zeros(pad_hi, y_h.dtype)])
            windows = np.lib.stride_tricks.sliding_window_view(padded, len(k))[lo + pad_lo]
            out[i] = np.abs(windows @ k) * self.scale[i]
        return out

    def column(self, y_h: np.ndarray, t: int) -> np.ndarray:
        """|CQT| of column t: every bin centred at sample (t - d_k) * hop, zeros outside the song."""
        out = np.empty(len(self.kernels))
        for i, (k, s, d) in enumerate(zip(self.kernels, self.starts, self.delays)):
            lo = (t - d) * self.hop + s
            hi = lo + len(k)
            a, b = max(lo, 0), min(hi, len(y_h))
            out[i] = 0.0 if a >= b else np.abs(np.dot(k[a - lo:b - lo], y_h[a:b])) * self.scale[i]
        return out


def sorted_percentile(values: np.ndarray, q: float) -> float:
    """np.percentile (linear interpolation) of an already sorted array, without sorting it again."""
    pos = q / 100 * (len(values) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(values) - 1)
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


class Buffer:
    """A float32 array that grows at the end in amortized constant time (capacity doubles)."""

    def __init__(self, n: int = 0):
        self.data = np.zeros(max(n, 1024), dtype=np.float32)
        self.n = n

    def grow_to(self, n: int) -> None:
        if n > len(self.data):
            data = np.zeros(max(n, 2 * len(self.data)), dtype=np.float32)
            data[:self.n] = self.data[:self.n]
            self.data = data
        self.n = max(self.n, n)

    def extend(self, x: np.ndarray) -> None:
        start = self.n
        self.grow_to(start + len(x))
        self.data[start:self.n] = x

    @property
    def view(self) -> np.ndarray:
        return self.data[:self.n]

    def __len__(self) -> int:
        return self.n


class TuningMedians:
    """features_v7_1.harmonic_medians for start_tuning's STFT of raw[:n], computed as the audio arrives.

    That STFT's frame j reads raw[j * 512 - 1,024 : j * 512 + 1,024] (zeros before 0 and after n). Every
    frame complete in the audio so far is computed on arrival (update), with its frequency median (that
    frame only) and its time median (the past frames). Medians only select values, so these are exactly
    the medians of the whole call. Called by start_tuning with the whole call's magnitudes, it keeps the
    columns whose magnitudes are bitwise the ones seen ahead and recomputes the others (the last frames,
    which read zeros after n): the result is harmonic_medians(mag, cfg), with most of its work done
    over the first tuning_seconds instead of on one block.
    """

    def __init__(self, cfg: dict, n: int):
        self.n = n
        self.width = cfg["audio"].get("harmonic_median_frames", MEDIAN_FRAMES)
        empty = np.zeros((HALF + 1, 0), dtype=np.float32)
        self.mag, self.H, self.P = empty, empty, empty

    def update(self, raw: np.ndarray) -> None:
        """The medians of every frame of raw[:n] whose samples have all arrived."""
        m = min(len(raw), self.n)
        j0, j1 = self.mag.shape[1], max(0, (m - HALF) // STFT_HOP + 1)  # frame j is complete if j * 512 + 1,024 <= m
        if j1 <= j0:
            return
        lo, hi = j0 * STFT_HOP - HALF, (j1 - 1) * STFT_HOP + HALF
        seg = raw[max(lo, 0):hi]
        if lo < 0:
            seg = np.concatenate([np.zeros(-lo, dtype=raw.dtype), seg])
        self._extend(np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=STFT_HOP, center=False)))

    def _extend(self, mag: np.ndarray) -> None:
        """Append frames: their frequency medians, and their time medians over the frames before them."""
        context = self.mag[:, max(0, self.mag.shape[1] - (self.width - 1)):]
        both = np.concatenate([context, mag], axis=1)
        H = median_filter(both, size=(1, self.width), origin=(0, self.width // 2), mode="constant", cval=0.0)
        P = median_filter(mag, size=(MEDIAN_FRAMES, 1), mode="reflect")
        self.mag = np.concatenate([self.mag, mag], axis=1)
        self.H = np.concatenate([self.H, H[:, context.shape[1]:]], axis=1)
        self.P = np.concatenate([self.P, P], axis=1)

    def __call__(self, mag: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
        """harmonic_medians(mag, cfg) for start_tuning's whole STFT: the frames seen ahead, the rest now."""
        k = min(self.mag.shape[1], mag.shape[1])
        same = (mag[:, :k] == self.mag[:, :k]).all(axis=0)
        keep = k if same.all() else int(np.argmin(same))       # the first frame that differs, and all after it
        self.mag, self.H, self.P = self.mag[:, :keep], self.H[:, :keep], self.P[:, :keep]
        self._extend(mag[:, keep:])
        return self.H, self.P


class StreamingFeatures:
    """V7.1's causal log-CQT, block by block."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        q, a = cfg["cqt"], cfg["audio"]
        self.hop, self.sr = q["hop_length"], a["sr"]
        self.width = a.get("harmonic_median_frames", MEDIAN_FRAMES)
        self.margin = a["harmonic_margin"]
        self.window = librosa.filters.get_window("hann", N_FFT, fftbins=True).astype(np.float32)
        self.raw = Buffer()                                     # the audio so far
        self.mags = []                                          # the last STFT magnitude frames (time median)
        self.acc = Buffer(HALF)                                 # overlap-add buffer, padded coordinates
        self.norm = Buffer(HALF)                                # sum of squared windows, padded coordinates
        self.y_h = Buffer()                                     # final harmonic samples
        self.frames = 0                                         # STFT frames computed
        self.d0 = features_v7_1.bin_delays(cfg, 0.0)
        self.kernels = Kernels(cfg, 0.0, self.d0)
        self.tuned = None
        self.n_tuning = int(q["tuning_seconds"] * self.sr)
        self.tuning_medians = TuningMedians(cfg, self.n_tuning)
        self.columns = 0                                        # log-CQT columns emitted
        self.sorted = np.zeros(0)                               # every CQT value so far, sorted
        self.comp = cfg["compression"]

    # ---------------------------------------------------------------- harmonic part
    def _segment(self, j: int) -> np.ndarray:
        """The 2,048 samples of STFT frame j (centred at sample j * 512), zeros before and after the audio."""
        lo = j * STFT_HOP - HALF
        out = np.zeros(N_FFT, dtype=np.float32)
        a, b = max(lo, 0), min(lo + N_FFT, len(self.raw))
        if a < b:
            out[a - lo:b - lo] = self.raw.view[a:b]
        return out

    def _stft_frame(self, j: int) -> None:
        """STFT frame j, its mask (time median over the past frames, frequency median), and its overlap-add."""
        S = scipy.fft.rfft(self.window * self._segment(j))
        mag = np.abs(S)
        self.mags = (self.mags + [mag])[-self.width:]
        past = np.stack(self.mags + [np.zeros_like(mag)] * (self.width - len(self.mags)))
        H = np.median(past, axis=0)
        P = median_filter(mag[:, None], size=(MEDIAN_FRAMES, 1), mode="reflect")[:, 0]
        mask = librosa.util.softmask(H, self.margin * P, power=2.0, split_zeros=False)
        y = self.window * scipy.fft.irfft(S * mask, n=N_FFT).astype(np.float32)
        end = j * STFT_HOP + N_FFT
        self.acc.grow_to(end)
        self.norm.grow_to(end)
        self.acc.data[j * STFT_HOP:end] += y
        self.norm.data[j * STFT_HOP:end] += self.window ** 2

    def _harmonic(self, total: int | None) -> None:
        """Compute every STFT frame the audio allows and finalize the harmonic samples no later frame touches.

        total: the song's length once it has ended (flush), else None.
        """
        if total is None:
            while self.frames * STFT_HOP + HALF <= len(self.raw):          # frame j reads up to j * 512 + 1,024
                self._stft_frame(self.frames)
                self.frames += 1
            final = max(0, min(self.frames * STFT_HOP - HALF, len(self.raw)))   # no later frame reaches these
        else:
            while self.frames < 1 + total // STFT_HOP:
                self._stft_frame(self.frames)
                self.frames += 1
            final = total
        start = len(self.y_h)
        if start < final:
            acc = self.acc.data[HALF + start:HALF + final]
            norm = self.norm.data[HALF + start:HALF + final]
            out = np.where(norm > np.finfo(np.float32).tiny, acc / np.where(norm > 0, norm, 1), acc)
            self.y_h.extend(out.astype(np.float32))

    # ---------------------------------------------------------------- CQT, level, log
    def _maybe_tune(self) -> None:
        """Once more than tuning_seconds of audio have arrived: the tuning estimated on them (start_tuning)."""
        if self.tuned is None and len(self.raw) > self.n_tuning:
            tuning, _ = features_v7_1.start_tuning(self.raw.view, self.cfg, self.tuning_medians)  # raw[:n_tuning]
            self.tuned = Kernels(self.cfg, tuning, np.maximum(self.d0, features_v7_1.bin_delays(self.cfg, tuning)))

    def _columns(self, n_total: int | None) -> list[np.ndarray]:
        """Every column whose harmonic samples (and tuning decision) are known; n_total: the song's length at flush."""
        out = []
        last = (1 + n_total // self.hop) if n_total is not None else None
        first_tuned = int(np.ceil(self.n_tuning / self.hop)) - 1    # as features_v7_1.start_tuning
        while last is None or self.columns < last:
            t = self.columns
            if t >= first_tuned and self.tuned is None and n_total is None:
                break                                           # not yet known whether the song passes 10 s
            k = self.tuned if (t >= first_tuned and self.tuned is not None) else self.kernels
            if n_total is None and k.needs(t) > len(self.y_h):
                break
            C = k.column(self.y_h.view, t)
            self.sorted = np.insert(self.sorted, np.searchsorted(self.sorted, np.sort(C)), np.sort(C))
            level = max(sorted_percentile(self.sorted, self.comp["level_percentile"]), 1e-10)
            out.append(np.log1p(self.comp["gamma"] * C / level).astype(np.float32))
            self.columns += 1
        return out

    # ---------------------------------------------------------------- public
    def step(self, block: np.ndarray) -> list[np.ndarray]:
        """Add the next block of samples; return every log-CQT column that has become computable."""
        self.raw.extend(np.asarray(block, dtype=np.float32))
        self._harmonic(None)
        if self.tuned is None:
            self.tuning_medians.update(self.raw.view)
        self._maybe_tune()
        return self._columns(None)

    def flush(self) -> list[np.ndarray]:
        """The song has ended: the remaining columns, with silence after the last sample."""
        n = len(self.raw)
        self._harmonic(n)
        self._maybe_tune()
        return self._columns(n)


def offline_log_cqt(y: np.ndarray, cfg: dict) -> np.ndarray:
    """The same direct computation on a whole file: features_v7_1's harmonic part and tuning, the direct CQT."""
    y_h = features_v7_1.causal_harmonic(y, cfg["audio"]["sr"], cfg)
    d0 = features_v7_1.bin_delays(cfg, 0.0)
    hop = cfg["cqt"]["hop_length"]
    n = 1 + len(y) // hop
    base = Kernels(cfg, 0.0, d0)
    C = base.all_columns(y_h, n)
    est = features_v7_1.start_tuning(y, cfg)
    if est is not None:
        tuning, first = est
        tuned = Kernels(cfg, tuning, np.maximum(d0, features_v7_1.bin_delays(cfg, tuning)))
        C[:, first:] = tuned.all_columns(y_h, n)[:, first:]
    comp = cfg["compression"]
    level = features_v7_1.running_level(C, comp["level_percentile"])
    return np.log1p(comp["gamma"] * C / level[None, :]).astype(np.float32)
