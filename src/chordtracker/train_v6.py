"""Component 16 (V6): training the CRNN on excerpts, as plain functions.

    run_epoch(model, opt, songs, rng, t)      one pass: every song tiled into excerpts, random order and shifts
    song_posteriors(model, song)              (24, n_frames) P(chord | frame) for a whole song, no shift
    frame_accuracy(model, songs)              share of target frames whose most likely class is right
    train(train_songs, val_sets, cfg)         epochs until val stops improving -> best model + history
    save(model, path) / load(path, ...)       weights only

V5's loop with excerpts instead of frames. One epoch: data_v6.excerpt_starts tiles every training
song into back-to-back excerpts of `excerpt_frames` frames (from a new random offset each epoch), so
every frame with a target is seen once, as in V5. Each excerpt gets one random shift k in -5..+6 for
all its frames. A batch is `batch_excerpts` excerpts (8 x 128 = 1,024 frames, V5 used 256 frames);
the loss is the cross-entropy over every frame with a target (-1 is ignored), Adam step.

Validation and early stopping are V5's: after each epoch, the frame accuracy of each val set with
every song read as one excerpt (k = 0, dropout off); the score is the mean over the val sets, the best
epoch's weights are kept, and training stops after `patience` epochs without a better score.

Reproducibility: cfg["train"]["seed"] fixes the weight init, the tiling offsets and the shifts; some
MPS kernels are not bit-exact, so two runs can differ in the last digits.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from chordtracker import data_v5, data_v6
from chordtracker.data_v5 import Song
from chordtracker.model_v5 import device
from chordtracker.model_v6 import ChordCRNN, frame_posteriors
from chordtracker.train_v5 import epochs_since_best


def run_epoch(model: nn.Module, opt: torch.optim.Optimizer, songs: list[Song], rng: np.random.Generator,
              t: dict) -> float:
    """One pass over this epoch's excerpts in a random order, one random shift each; returns the mean loss.

    t: excerpt_frames, batch_excerpts.
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
        x, y = data_v6.make_batch(songs, starts[pick], shifts[pick], length)
        opt.zero_grad()
        logits = model(x.to(dev))
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.to(dev).reshape(-1),
                                           ignore_index=-1)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def song_posteriors(model: nn.Module, song: Song) -> np.ndarray:
    """(24, n_frames) frame posteriors of a whole song read as one excerpt, unshifted (k = 0), eval mode."""
    n = song.cqt.shape[1]
    strip = torch.from_numpy(data_v6.excerpt(song, 0, 0, n))[None, None]
    return frame_posteriors(model, strip).numpy()


def frame_accuracy(model: nn.Module, songs: list[Song]) -> float:
    """Share of frames with a target whose most likely class is the target, pooled over the songs."""
    right = total = 0
    for song in songs:
        keep = song.targets >= 0
        right += int((song_posteriors(model, song).argmax(axis=0)[keep] == song.targets[keep]).sum())
        total += int(keep.sum())
    return right / total


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict) -> tuple[nn.Module, list[dict]]:
    """Train from scratch with early stopping; returns the best epoch's model and one history row per epoch.

    cfg["train"]: seed, lr, max_epochs, patience, dropout, excerpt_frames, batch_excerpts, bidirectional.
    """
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = ChordCRNN(bidirectional=t["bidirectional"], dropout=t["dropout"]).to(device())
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])

    history, scores, best = [], [], None
    for epoch in range(1, t["max_epochs"] + 1):
        began = time.time()
        loss = run_epoch(model, opt, train_songs, rng, t)
        accuracy = {name: frame_accuracy(model, songs) for name, songs in val_sets.items()}
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


def save(model: nn.Module, path) -> None:
    """Weights only (state_dict), on the CPU so the file loads on any machine."""
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)


def load(path, bidirectional: bool = True, dropout: float = 0.5) -> nn.Module:
    """A ChordCRNN with the saved weights, on device(), in eval mode."""
    model = ChordCRNN(bidirectional=bidirectional, dropout=dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device()).eval()
