"""Component 40 (V13): V11's stage B with a played-bass loss.

    played_bass(notes, n_frames, sr, hop)    (n,) lowest bass note sounding at each frame time: 0-11, or
                                             12 (no bass note); from midi.bass_notes of an aligned MIDI file
    excerpt_bass(bass, start, k, length)     the excerpt's bass targets k semitones higher; -1 = no target
    run_epoch(model, opt, songs, rng, t, teachers, bass)
    train(train_songs, val_sets, cfg, n_classes, teachers, init, bass)

Loss = train_v11's (chord CE on the labels + kl_weight * KL to the V8 ensemble) + bass_weight * CE of the
bass head against the played bass, on the frames that have a bass target. A song without a MIDI bass part
(GuitarSet, ChoCo, RWC-P songs whose MIDI has no bass track) has bass None: every frame is masked. The
pitch shift moves the bass with the audio: class c becomes (c + k) % 12, "no bass" stays "no bass". The
excerpts, their order and their shifts are train_v11's for the same seed (the bass targets draw nothing).
"""
import numpy as np
import torch
from torch import nn

from chordtracker import data_v5, data_v6, data_v7_2, model_v13
from chordtracker.data_v5 import Song
from chordtracker.model_v5 import device
from chordtracker.train_v7_2 import frame_loss
from chordtracker.train_v11 import distill_loss, fit, inside_song, teacher_log_posteriors


def played_bass(notes: list[tuple[float, float, int, int]], n_frames: int, sr: int, hop: int) -> np.ndarray:
    """(n_frames,) pitch class of the lowest bass note sounding at each frame's time (t * hop / sr), 12 if none.

    Example: notes E1 0.0-0.5 s and A1 0.4-1.0 s, frames at 0.0, 0.45, 0.9, 1.2 s -> [4, 4, 9, 12].
    """
    times = np.arange(n_frames) * hop / sr
    out = np.full(n_frames, model_v13.NO_BASS, dtype=int)
    low = np.full(n_frames, np.inf)
    for start, end, pitch, _v in notes:
        on = (times >= start) & (times < end) & (pitch < low)
        low[on] = pitch
        out[on] = pitch % 12
    return out


def excerpt_bass(bass: np.ndarray | None, start: int, k: int, length: int) -> np.ndarray:
    """(length,) bass targets of frames start .. start + length - 1, k semitones higher; -1 outside the song
    or for a song without bass (None); "no bass" (12) stays 12."""
    y = np.full(length, -1, dtype=int)
    if bass is None:
        return y
    lo, hi = max(start, 0), min(start + length, len(bass))
    if lo < hi:
        y[lo - start:hi - start] = bass[lo:hi]
    pitched = (y >= 0) & (y < model_v13.NO_BASS)
    return np.where(pitched, (y + k) % 12, y)


def run_epoch(model: nn.Module, opt: torch.optim.Optimizer, songs: list[Song], rng: np.random.Generator, t: dict,
              teachers: list[nn.Module] | None, bass: list[np.ndarray | None]) -> tuple[float, float]:
    """train_v11.run_epoch plus bass_weight * the bass CE; returns the mean chord CE and the mean KL."""
    model.train()
    dev = next(model.parameters()).device
    length, w, wb = t["excerpt_frames"], t.get("kl_weight", 0), t["bass_weight"]
    starts = data_v6.excerpt_starts(songs, length, rng)
    order = rng.permutation(len(starts))
    shifts = rng.choice(list(data_v5.SHIFTS), size=len(starts))
    ces, kls = [], []
    for first in range(0, len(order), t["batch_excerpts"]):
        pick = order[first:first + t["batch_excerpts"]]
        x, y = data_v7_2.make_batch(songs, starts[pick], shifts[pick], length, t["input_lookahead"])
        yb = np.stack([excerpt_bass(bass[i], s, k, length) for (i, s), k in zip(starts[pick], shifts[pick])])
        x, y, yb = x.to(dev), y.to(dev), torch.from_numpy(yb).long().to(dev)
        opt.zero_grad()
        logits, bass_logits = model.heads(x)
        ce = frame_loss(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce
        if (yb >= 0).any():
            loss = loss + wb * frame_loss(bass_logits.reshape(-1, model_v13.N_BASS), yb.reshape(-1))
        if w:
            kl = distill_loss(logits, teacher_log_posteriors(teachers, x), inside_song(songs, starts[pick], length).to(dev))
            loss = loss + w * kl
            kls.append(kl.item())
        loss.backward()
        opt.step()
        ces.append(ce.item())
    return float(np.mean(ces)), float(np.mean(kls)) if kls else 0.0


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict, n_classes: int,
          teachers: list[nn.Module] | None, init: dict, bass: list[np.ndarray | None]) -> tuple[nn.Module, list[dict]]:
    """train_v11.train from `init` (a chord model's state, e.g. V11's stage A) with the bass head and loss;
    early stopping on the chord head's val frame accuracy, as V8."""
    t = cfg["train"]
    if len(bass) != len(train_songs):
        raise ValueError(f"{len(bass)} bass targets for {len(train_songs)} songs")
    for m in teachers or []:
        m.to(device()).eval().requires_grad_(False)
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = model_v13.from_chord_model(init, n_classes, t["dropout"]).to(device())
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    return fit(model, lambda: run_epoch(model, opt, train_songs, rng, t, teachers, bass), val_sets, t)
