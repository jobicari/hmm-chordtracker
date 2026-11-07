"""Component 32 (V8): training the causal CRNN on a richer vocabulary, as plain functions.

train_v7_2's loop, epoch and frame accuracy, with model_v8's output size: the songs' targets are
data_v8 classes (quality * 12 + root), and data_v6's excerpts shift them within their quality block
(data_v5.shift_targets), so nothing else changes.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from chordtracker.data_v5 import Song
from chordtracker.model_v5 import device
from chordtracker.model_v8 import ChordCRNN
from chordtracker.train_v5 import epochs_since_best
from chordtracker.train_v6 import load, save  # noqa: F401  (same weights format as V6)
from chordtracker.train_v7_2 import frame_accuracy, run_epoch, song_posteriors  # noqa: F401


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict,
          n_classes: int) -> tuple[nn.Module, list[dict]]:
    """Train a causal CRNN with n_classes outputs from scratch with early stopping, as train_v7_2.train.

    cfg["train"]: seed, lr, max_epochs, patience, dropout, excerpt_frames, batch_excerpts, input_lookahead.
    """
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = ChordCRNN(n_classes, bidirectional=False, dropout=t["dropout"]).to(device())
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
