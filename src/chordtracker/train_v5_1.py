"""V5.1: training the 25-class acoustic model. train_v5's loop, with the two pieces that know the classes swapped.

    make_batch(songs, examples, shifts)      as train_v5.make_batch, targets shifted by data_v5_1 (N stays N)
    run_epoch(model, opt, songs, ...)        as train_v5.run_epoch, with this make_batch
    train(train_songs, val_sets, cfg)        as train_v5.train, starting from model_v5_1.ChordCNNWithN
    load(path, dropout)                      a ChordCNNWithN with the saved weights

Why not call train_v5 as it is: it shifts targets with data_v5.shift_targets, which reads class 24 as
"minor, root 0" and would move N to 24 + k, a class that does not exist. Everything that does not
depend on the classes is train_v5's, unchanged: training_frames, song_posteriors (which returns
(25, n_frames) here), frame_accuracy, epochs_since_best, save.

Frame accuracy on val now includes the N frames of RWC-P val (about 2% of them), so early stopping
sees the new class too. GuitarSet val has no N frames, so its accuracy is still chords only.
"""
import copy

import numpy as np
import torch
from torch import nn

from chordtracker import data_v5, data_v5_1
from chordtracker.data_v5 import Song
from chordtracker.model_v5 import device
from chordtracker.model_v5_1 import ChordCNNWithN
from chordtracker.train_v5 import epochs_since_best, frame_accuracy, training_frames


def make_batch(songs: list[Song], examples: np.ndarray, shifts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Patches (B, 1, 144, 15) of the examples, each shifted by its k, and the targets shifted with them."""
    x = np.stack([data_v5.patch(songs[i], t, k) for (i, t), k in zip(examples, shifts)])[:, None]
    y = np.array([data_v5_1.shift_targets(songs[i].targets[t:t + 1], k)[0] for (i, t), k in zip(examples, shifts)])
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


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict) -> tuple[nn.Module, list[dict]]:
    """Train from scratch with early stopping, as train_v5.train; returns the best epoch's model and the history."""
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = ChordCNNWithN(dropout=t["dropout"]).to(device())
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


def load(path, dropout: float = 0.5) -> nn.Module:
    """A ChordCNNWithN with the saved weights, on device(), in eval mode."""
    model = ChordCNNWithN(dropout=dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device()).eval()
