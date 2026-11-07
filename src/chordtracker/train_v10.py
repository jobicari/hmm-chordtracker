"""Component 37 (V10): V8's training with three options, each off by default (then it is train_v8.train).

    cfg["train"]["early_stop"]     "frame_accuracy" (V8): keep the epoch with the best mean frame accuracy
                                   of the val sets; "sevenths": the best mean sevenths WCSR of the network
                                   alone (frame argmax) over the val sets, the measure the report uses
    cfg["train"]["aux_weight"]     0 (V8); w > 0 trains model_v10's root, quality and bass heads too:
                                   loss = chord CE + w * (root CE + quality CE + bass CE)
    cfg["train"]["augment"]        absent or all 0 (V8); gain_db, tilt_db, noise: extra input augmentation

    bass_targets(annotation, times)       (n,) bass pitch class at each time, -1 for N, X or no chord
    augment(x, rng, a)                    the batch's log-CQT strips with a random gain, tilt and noise
    val_sevenths(model, scored, lookahead)  {set: sevenths WCSR of the network alone}
    train(train_songs, val_sets, cfg, n_classes, bass=None, scored=None)

Heads. Root and quality come from the chord target (class = quality * 12 + root), so they are trained on
the same frames as the chord; bass comes from the Harte label through mir_eval.chord.encode (bass pitch
class = root + bass interval), so it also has a target on chords outside the vocabulary (C:dim/b3). Every
excerpt is shifted by k semitones: the chord's root and the bass move by k together, the quality stays.

Augmentation. The cached feature is x = log(1 + 10 C / level). Gain and tilt scale C / level and keep
the compression: x' = log(1 + (e^x - 1) * g_b), with g_b = 10 ** ((gain + tilt * (b / 143 - 0.5)) / 20)
for row b of the 144-row strip, gain uniform in +/- gain_db and tilt (dB from the lowest row to the
highest) uniform in +/- tilt_db, both drawn per excerpt. Noise adds N(0, noise^2) to every value of the
excerpt's columns inside the song, clipped at 0. A separate generator draws them, so the excerpts, their
order and their shifts are V8's for the same seed.
"""
import copy
import time

import mir_eval
import numpy as np
import torch
from torch import nn

from chordtracker import data_v5, data_v6, data_v7_2, frame_score
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5 import Song
from chordtracker.model_v5 import device
from chordtracker.model_v8 import ChordCRNN
from chordtracker.model_v10 import ChordCRNNHeads
from chordtracker.train_v5 import epochs_since_best
from chordtracker.train_v7_2 import frame_accuracy, frame_loss, song_posteriors

AUGMENT_SEED_OFFSET = 1000          # the augmentation generator's seed is the training seed + this


def bass_targets(annotation: ChordAnnotation, times: np.ndarray) -> np.ndarray:
    """(n,) the bass pitch class of the chord sounding at each time; -1 for N, X, bad labels and outside.

    Examples: 'C:maj' -> 0, 'C:maj/3' -> 4, 'A:min7/b7' -> 7, 'D:dim' -> 2, 'N' -> -1.
    """
    classes = []
    for label in annotation.labels:
        try:
            root, _, bass = mir_eval.chord.encode(label)
        except mir_eval.chord.InvalidChordException:
            root = -1
        classes.append(-1 if root < 0 else int((root + bass) % 12))
    classes = np.array(classes, dtype=int)
    segment = np.searchsorted(annotation.intervals[:, 0], times, side="right") - 1
    inside = (segment >= 0) & (times < annotation.intervals[np.clip(segment, 0, None), 1])
    return np.where(inside, classes[np.clip(segment, 0, None)], -1)


def excerpt_bass(bass: np.ndarray, start: int, k: int, length: int) -> np.ndarray:
    """(length,) bass targets of frames start .. start + length - 1, k semitones higher; -1 stays -1."""
    y = np.full(length, -1, dtype=int)
    lo, hi = max(start, 0), min(start + length, len(bass))
    if lo < hi:
        y[lo - start:hi - start] = bass[lo:hi]
    return np.where(y < 0, -1, (y + k) % 12)


def augment(x: np.ndarray, rng: np.random.Generator, a: dict) -> np.ndarray:
    """(B, 1, rows, cols) strips with gain, tilt and noise as in the module docstring; a: gain_db, tilt_db, noise."""
    b, _, rows, _ = x.shape
    gain = rng.uniform(-1, 1, b) * a.get("gain_db", 0.0)
    tilt = rng.uniform(-1, 1, b) * a.get("tilt_db", 0.0)
    db = gain[:, None] + tilt[:, None] * (np.arange(rows)[None, :] / (rows - 1) - 0.5)       # (B, rows)
    g = (10.0 ** (db / 20.0))[:, None, :, None]
    out = np.log1p(np.expm1(x) * g)
    if a.get("noise", 0.0):
        inside = (x != 0).any(axis=2, keepdims=True)                                         # columns in the song
        out = np.maximum(out + rng.normal(0.0, a["noise"], x.shape) * inside, 0.0)
    return out.astype(np.float32)


def augmenting(t: dict) -> bool:
    a = t.get("augment") or {}
    return any(a.get(k, 0) for k in ("gain_db", "tilt_db", "noise"))


def run_epoch(model: nn.Module, opt: torch.optim.Optimizer, songs: list[Song], rng: np.random.Generator,
              t: dict, bass: list[np.ndarray] | None = None, aug_rng: np.random.Generator | None = None) -> float:
    """train_v7_2.run_epoch with the optional heads loss and augmentation; returns the mean loss."""
    model.train()
    dev = next(model.parameters()).device
    length, w = t["excerpt_frames"], t.get("aux_weight", 0)
    starts = data_v6.excerpt_starts(songs, length, rng)
    order = rng.permutation(len(starts))
    shifts = rng.choice(list(data_v5.SHIFTS), size=len(starts))
    losses = []
    for first in range(0, len(order), t["batch_excerpts"]):
        pick = order[first:first + t["batch_excerpts"]]
        x, y = data_v7_2.make_batch(songs, starts[pick], shifts[pick], length, t["input_lookahead"])
        if aug_rng is not None:
            x = torch.from_numpy(augment(x.numpy(), aug_rng, t["augment"]))
        x, y = x.to(dev), y.to(dev).reshape(-1)
        opt.zero_grad()
        if w:
            chord, root, quality, low = model.heads(x)
            yb = torch.from_numpy(np.stack([excerpt_bass(bass[i], s, k, length)
                                            for (i, s), k in zip(starts[pick], shifts[pick])])).long().to(dev)
            has = y >= 0
            yr, yq = torch.where(has, y % 12, -1), torch.where(has, y // 12, -1)
            loss = (frame_loss(chord.reshape(-1, chord.shape[-1]), y)
                    + w * (frame_loss(root.reshape(-1, 12), yr) + frame_loss(quality.reshape(-1, quality.shape[-1]), yq)
                           + frame_loss(low.reshape(-1, 12), yb.reshape(-1))))
        else:
            logits = model(x)
            loss = frame_loss(logits.reshape(-1, logits.shape[-1]), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def val_sevenths(model: nn.Module, scored: dict[str, list[tuple[Song, np.ndarray, np.ndarray]]],
                 lookahead: int) -> dict[str, float]:
    """{set: pooled sevenths WCSR of the frame argmax}; scored[set] holds (song, M, S) frame_score tables."""
    out = {}
    for name, items in scored.items():
        matched = total = 0.0
        for song, M, S in items:
            m, s = frame_score.score_path(M, S, song_posteriors(model, song, lookahead).argmax(axis=0))
            matched, total = matched + m, total + s
        out[name] = matched / total
    return out


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict, n_classes: int,
          bass: list[np.ndarray] | None = None,
          scored: dict[str, list[tuple[Song, np.ndarray, np.ndarray]]] | None = None) -> tuple[nn.Module, list[dict]]:
    """train_v8.train with the options above. bass: one bass_targets array per train song (aux_weight > 0);
    scored: the val sets with their sevenths tables (early_stop "sevenths", and logged whenever given)."""
    t = cfg["train"]
    stop_on = t.get("early_stop", "frame_accuracy")
    if stop_on not in ("frame_accuracy", "sevenths"):
        raise ValueError(f"unknown early_stop {stop_on!r}")
    if stop_on == "sevenths" and scored is None:
        raise ValueError("early_stop sevenths needs the val sets' sevenths tables")
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    aug_rng = np.random.default_rng(t["seed"] + AUGMENT_SEED_OFFSET) if augmenting(t) else None
    if t.get("aux_weight", 0):
        model = ChordCRNNHeads(n_classes, n_classes // 12, dropout=t["dropout"]).to(device())
    else:
        model = ChordCRNN(n_classes, bidirectional=False, dropout=t["dropout"]).to(device())
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])

    history, scores, best = [], [], None
    for epoch in range(1, t["max_epochs"] + 1):
        began = time.time()
        loss = run_epoch(model, opt, train_songs, rng, t, bass, aug_rng)
        accuracy = {name: frame_accuracy(model, songs, t["input_lookahead"]) for name, songs in val_sets.items()}
        row = {"epoch": epoch, "train_loss": loss, **{f"val_{n}": a for n, a in accuracy.items()},
               "val_mean": float(np.mean(list(accuracy.values())))}
        if scored is not None:
            sev = val_sevenths(model, scored, t["input_lookahead"])
            row.update({f"val_sevenths_{n}": s for n, s in sev.items()}, val_sevenths_mean=float(np.mean(list(sev.values()))))
        row["seconds"] = time.time() - began
        history.append(row)
        scores.append(row["val_mean"] if stop_on == "frame_accuracy" else row["val_sevenths_mean"])
        print(f"epoch {epoch:3d}  loss {loss:.3f}  " + "  ".join(f"{n} {100 * a:.2f}%" for n, a in accuracy.items())
              + (f"  sevenths {100 * row['val_sevenths_mean']:.2f}%" if scored is not None else "")
              + f"  {row['seconds']:.0f} s", flush=True)
        if epochs_since_best(scores) == 0:
            best = copy.deepcopy(model.state_dict())
        elif epochs_since_best(scores) >= t["patience"]:
            break
    model.load_state_dict(best)
    return model, history
