"""Component 24 (V7.2): training the causal CRNN with an input look-ahead, as plain functions.

train_v7_1's loop with data_v7_2's excerpts: cfg["train"]["input_lookahead"] frames of input past
each frame (0 = V7.1). Everything else is V6's and V7.1's.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from chordtracker import data_v5, data_v6, data_v7_2
from chordtracker.data_v5 import Song
from chordtracker.model_v5 import device
from chordtracker.model_v6 import ChordCRNN, frame_posteriors
from chordtracker.train_v5 import epochs_since_best
from chordtracker.train_v6 import load, save  # noqa: F401  (same weights format as V6)


def run_epoch(model: nn.Module, opt: torch.optim.Optimizer, songs: list[Song], rng: np.random.Generator,
              t: dict, weight: torch.Tensor | None = None) -> float:
    """One pass over this epoch's excerpts in a random order, one random shift each; returns the mean loss.

    weight: one loss weight per class (train_v8.class_weights), or None for the plain cross-entropy.
    """
    model.train()
    dev = next(model.parameters()).device
    length = t["excerpt_frames"]
    starts = data_v6.excerpt_starts(songs, length, rng)
    order = rng.permutation(len(starts))
    shifts = rng.choice(list(data_v5.SHIFTS), size=len(starts))
    losses = []
    for first in range(0, len(order), t["batch_excerpts"]):
        pick = order[first:first + t["batch_excerpts"]]
        x, y = data_v7_2.make_batch(songs, starts[pick], shifts[pick], length, t["input_lookahead"])
        opt.zero_grad()
        logits = model(x.to(dev))
        loss = frame_loss(logits.reshape(-1, logits.shape[-1]), y.to(dev).reshape(-1), weight)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def frame_loss(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy of (n, n_classes) logits against (n,) targets, frames with target -1 left out.

    Without weight: the mean over the frames with a target. With weight: each frame's loss times the weight
    of its target class, summed and divided by the number of frames with a target (not by the summed
    weights, as torch's weighted mean does), so weights with a frame-weighted mean of 1 keep the loss scale.
    """
    if weight is None:
        return nn.functional.cross_entropy(logits, y, ignore_index=-1)
    total = nn.functional.cross_entropy(logits, y, weight=weight.to(logits.device), ignore_index=-1, reduction="sum")
    return total / (y >= 0).sum().clamp(min=1)


def song_posteriors(model: nn.Module, song: Song, lookahead: int) -> np.ndarray:
    """(24, n_frames) frame posteriors of a whole song read as one excerpt, unshifted, eval mode."""
    n = song.cqt.shape[1]
    strip = torch.from_numpy(data_v7_2.excerpt(song, 0, 0, n, lookahead))[None, None]
    return frame_posteriors(model, strip).numpy()


def frame_accuracy(model: nn.Module, songs: list[Song], lookahead: int) -> float:
    """Share of frames with a target whose most likely class is the target, pooled over the songs."""
    right = total = 0
    for song in songs:
        keep = song.targets >= 0
        right += int((song_posteriors(model, song, lookahead).argmax(axis=0)[keep] == song.targets[keep]).sum())
        total += int(keep.sum())
    return right / total


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict) -> tuple[nn.Module, list[dict]]:
    """Train a causal CRNN from scratch with early stopping; the best epoch's model and one history row per epoch.

    cfg["train"]: seed, lr, max_epochs, patience, dropout, excerpt_frames, batch_excerpts, input_lookahead.
    """
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = ChordCRNN(bidirectional=False, dropout=t["dropout"]).to(device())
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])

    history, scores, best = [], [], None
    for epoch in range(1, t["max_epochs"] + 1):
        began = time.time()
        loss = run_epoch(model, opt, train_songs, rng, t)
        accuracy = {name: frame_accuracy(model, songs, t["input_lookahead"]) for name, songs in val_sets.items()}
        scores.append(float(np.mean(list(accuracy.values()))))
        history.append({"epoch": epoch, "train_loss": loss, **{f"val_{n}": a for n, a in accuracy.items()},
                        "val_mean": scores[-1], "seconds": time.time() - began})
        print(f"epoch {epoch:3d}  loss {loss:.3f}  " + "  ".join(f"{n} {100 * a:.2f}%" for n, a in accuracy.items())
              + f"  {history[-1]['seconds']:.0f} s", flush=True)
        if epochs_since_best(scores) == 0:
            best = copy.deepcopy(model.state_dict())
        elif epochs_since_best(scores) >= t["patience"]:
            break
    model.load_state_dict(best)
    return model, history
