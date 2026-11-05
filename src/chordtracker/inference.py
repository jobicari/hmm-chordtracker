"""The frozen V6 model (bidirectional CRNN, seed 1, reports/v6.md) as a library: chord posteriors of a
whole recording, for projects that install this package.

    load_v6()                                  the model with the packaged weights, on the CPU, eval mode
    posteriors(path, phases=1, model=None, tuning=None)
                                               (times, 24 x n posteriors), CLASSES order

Everything a caller needs ships in the package: SETTINGS holds V6's audio, CQT and compression settings
(configs/v6.yaml on top of v2.yaml, checked by tests/test_inference.py) and weights/ the frozen weights
(results/v6/bidir_seed1/model.pt). The whole song is read at once, as train_v6.song_posteriors reads a
validation song: offline, both directions of the GRU.

phases: V6's frames are 93 ms apart (hop 2048). With phases = p, the song is read p times, delayed by
0, 1/p, ..., (p - 1)/p of a hop, and the frames interleaved: one every hop / p, each computed from
exactly the input V6 was trained on. Finer sampling, not finer resolution: the model's own timing does
not change.

tuning: the recording's deviation from A440 in semitones, as the caller knows it (e.g. checked against
annotated chords); by default estimated on the semitone, +-0.5. V6's training features (features_v5)
estimate it on the CQT's quarter-tone bins, which wraps at +-25 cents: close for recordings within 25
cents of A440, but one 44 cents flat (Another Girl, Help!) read as 6 cents sharp, its notes between the
bins the model reads, and its chords a semitone low. Near a quarter tone off, the estimate itself may land
on the wrong semitone; a caller with the chords can try both.
"""
from functools import cache
from pathlib import Path

import librosa
import numpy as np
import torch

from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS, PATCH_FRAMES
from chordtracker.features import load_audio
from chordtracker.features_v2 import log_compress
from chordtracker.model_v6 import ChordCRNN, frame_posteriors

SETTINGS = {
    "audio": {"sr": 22050, "mono": True, "harmonic_margin": 4.0},
    "cqt": {"hop_length": 2048, "fmin": "F#0", "bins_per_octave": 24, "n_bins": 168},
    "compression": {"enabled": True, "level_percentile": 99, "gamma": 10},
}
V6_WEIGHTS = Path(__file__).parent / "weights" / "v6_bidir_seed1.pt"
_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CLASSES = [f"{root}:maj" for root in _ROOTS] + [f"{root}:min" for root in _ROOTS]


@cache
def load_v6() -> ChordCRNN:
    """The frozen V6 model on the CPU, in eval mode (loaded once per process)."""
    model = ChordCRNN(bidirectional=True)
    model.load_state_dict(torch.load(V6_WEIGHTS, map_location="cpu"))
    return model.eval()


def posteriors(
    path: Path | str, phases: int = 1, model: ChordCRNN | None = None, tuning: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """P(chord | frame) of a whole recording: times (n,) in seconds from the start, ascending, and the
    (24, n) posteriors in CLASSES order. Audio, CQT and compression as features_v5.log_cqt; `tuning` in
    semitones from A440, estimated on the semitone when None (see the module docstring).

    Raises ValueError when `phases` does not divide the hop.
    """
    q, sr = SETTINGS["cqt"], SETTINGS["audio"]["sr"]
    hop = q["hop_length"]
    if phases < 1 or hop % phases:
        raise ValueError(f"phases must divide the hop ({hop} samples), got {phases}")
    model = model or load_v6()
    y, sr = load_audio(path, SETTINGS)
    if tuning is None:
        tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    bins = tuning * q["bins_per_octave"] / 12  # librosa.cqt takes the tuning in fractions of its bins
    times, posts = [], []
    for p in range(phases):
        delay = p * hop // phases
        C = np.abs(librosa.cqt(np.concatenate([np.zeros(delay, dtype=y.dtype), y]), sr=sr, hop_length=hop,
                               fmin=librosa.note_to_hz(q["fmin"]), n_bins=q["n_bins"],
                               bins_per_octave=q["bins_per_octave"], tuning=bins))
        cqt = log_compress(C, SETTINGS["compression"]).astype(np.float32)
        n, margin = cqt.shape[1], PATCH_FRAMES // 2
        strip = np.zeros((PATCH_BINS, n + 2 * margin), dtype=np.float32)      # as data_v6.excerpt(song, 0, 0, n)
        strip[:, margin:margin + n] = cqt[CENTRE_ROW:CENTRE_ROW + PATCH_BINS]
        post = frame_posteriors(model, torch.from_numpy(strip)[None, None]).numpy()
        t = (np.arange(n) * hop - delay) / sr                                 # frame k of a pass delayed by `delay`
        keep = t >= 0
        times.append(t[keep])
        posts.append(post[:, keep])
    times, post = np.concatenate(times), np.concatenate(posts, axis=1)
    order = np.argsort(times, kind="stable")
    return times[order], post[:, order]
