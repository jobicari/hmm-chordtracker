"""Component 16 (V5): training the acoustic model, as plain functions.

    training_frames(songs)                    every (song, frame) with a target: the training examples
    make_batch(songs, examples, shifts)       (B, 1, 144, 15) patches + their shifted targets
    run_epoch(model, opt, songs, examples, rng, batch)   one pass over the examples, random order and shifts
    song_posteriors(model, song)              (24, n_frames) P(chord | frame) for a whole song, no shift
    frame_accuracy(model, songs)              share of target frames whose most likely class is right
    train(train_songs, val_sets, cfg)         epochs until val stops improving -> best model + history

One epoch: shuffle the ~160k training frames, give each a random shift k in -5..+6, and for every batch
of 256: cut the patches (data_v5.patch), forward, cross-entropy against the shifted targets, backward,
Adam step. After each epoch, the frame accuracy on each val set (no shift, dropout off).

Early stopping: the score of an epoch is the mean of the val sets' frame accuracies (RWC-P val and
GuitarSet val count equally, though RWC-P has 6x more frames). The best epoch's weights are kept;
training stops when `patience` epochs in a row have not beaten it. Frame accuracy, not the HMM's WCSR,
because it costs one forward pass and needs no beat tracking; the decoder is tuned afterwards.

Reproducibility: cfg["train"]["seed"] fixes the weight init, the shuffling and the shifts. On the
Mac's GPU some kernels are not bit-exact, so two runs can differ in the last digits.
"""
import copy

import numpy as np
import torch
from torch import nn

from chordtracker import data_v5
from chordtracker.data_v5 import Song
from chordtracker.model_v5 import ChordCNN, device, frame_posteriors


def training_frames(songs: list[Song]) -> np.ndarray:
    """(n, 2) array of (song index, frame index) for every frame with a target (not -1)."""
    return np.concatenate([np.stack([np.full(int((s.targets >= 0).sum()), i), np.flatnonzero(s.targets >= 0)], axis=1)
                           for i, s in enumerate(songs)])


def make_batch(songs: list[Song], examples: np.ndarray, shifts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Patches (B, 1, 144, 15) of the examples, each shifted by its k, and the targets shifted with them."""
    x = np.stack([data_v5.patch(songs[i], t, k) for (i, t), k in zip(examples, shifts)])[:, None]
    y = np.array([data_v5.shift_targets(songs[i].targets[t:t + 1], k)[0] for (i, t), k in zip(examples, shifts)])
    return torch.from_numpy(x), torch.from_numpy(y).long()


def run_epoch(model: nn.Module, opt: torch.optim.Optimizer, songs: list[Song], examples: np.ndarray,
              rng: np.random.Generator, batch: int) -> float:
    """One pass over the examples in a new random order, a new random shift each; returns the mean loss."""
    model.train()
    dev = next(model.parameters()).device
    order = rng.permutation(len(examples))
    shifts = rng.choice(list(data_v5.SHIFTS), size=len(examples))
    losses = []
    for start in range(0, len(order), batch):
        pick = order[start:start + batch]
        x, y = make_batch(songs, examples[pick], shifts[pick])
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x.to(dev)), y.to(dev))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def song_posteriors(model: nn.Module, song: Song) -> np.ndarray:
    """(24, n_frames) frame posteriors of a whole song, unshifted (k = 0), in eval mode."""
    patches = np.stack([data_v5.patch(song, t, 0) for t in range(song.cqt.shape[1])])[:, None]
    return frame_posteriors(model, torch.from_numpy(patches)).numpy()


def frame_accuracy(model: nn.Module, songs: list[Song]) -> float:
    """Share of frames with a target whose most likely class is the target, pooled over the songs."""
    right = total = 0
    for song in songs:
        keep = song.targets >= 0
        right += int((song_posteriors(model, song).argmax(axis=0)[keep] == song.targets[keep]).sum())
        total += int(keep.sum())
    return right / total


def epochs_since_best(scores: list[float]) -> int:
    """How many epochs have passed since the best score. Example: [0.5, 0.7, 0.6, 0.65] -> 2."""
    return len(scores) - 1 - int(np.argmax(scores))


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict) -> tuple[nn.Module, list[dict]]:
    """Train from scratch with early stopping; returns the best epoch's model and one history row per epoch.

    cfg["train"]: seed, lr, batch, max_epochs, patience, dropout.
    """
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = ChordCNN(dropout=t["dropout"]).to(device())
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    examples = training_frames(train_songs)

    history, scores, best = [], [], None
    for epoch in range(1, t["max_epochs"] + 1):
        loss = run_epoch(model, opt, train_songs, examples, rng, t["batch"])
        accuracy = {name: frame_accuracy(model, songs) for name, songs in val_sets.items()}
        scores.append(float(np.mean(list(accuracy.values()))))
        history.append({"epoch": epoch, "train_loss": loss, **{f"val_{n}": a for n, a in accuracy.items()},
                        "val_mean": scores[-1]})
        print(f"epoch {epoch:3d}  loss {loss:.3f}  " + "  ".join(f"{n} {100 * a:.2f}%" for n, a in accuracy.items()),
              flush=True)
        if epochs_since_best(scores) == 0:
            best = copy.deepcopy(model.state_dict())
        elif epochs_since_best(scores) >= t["patience"]:
            break
    model.load_state_dict(best)
    return model, history


def save(model: nn.Module, path) -> None:
    """Weights only (state_dict), on the CPU so the file loads on any machine."""
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)


def load(path, dropout: float = 0.5) -> nn.Module:
    """A ChordCNN with the saved weights, on device(), in eval mode."""
    model = ChordCNN(dropout=dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device()).eval()
