"""Component 32 (V8): training the causal CRNN on a richer vocabulary, as plain functions.

train_v7_2's loop, epoch and frame accuracy, with model_v8's output size: the songs' targets are
data_v8 classes (quality * 12 + root), and data_v6's excerpts shift them within their quality block
(data_v5.shift_targets), so nothing else changes.

    class_weights(targets, n_classes, alpha, floor)   one loss weight per class, from the train targets

Class weights (V9): weight_c = max(f_c, floor) ** -alpha, scaled so that sum_c f_c * weight_c = 1 (the
mean weight over the training frames is 1). f_c is the share of training frames of class c counted in all
12 transpositions (transitions_v8.training_prior): every excerpt is trained at a random one of the 12
shifts, so this pooled share per quality, not the root as annotated, is how often the loss sees each
class. The floor keeps a class that never occurs from an unbounded weight. alpha = 0 (the default) is the
unweighted V8 loss; cfg["train"]["class_weight_alpha"] and ["class_weight_floor"] set them.
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
from chordtracker.transitions_v8 import training_prior

CLASS_WEIGHT_FLOOR = 1e-4        # share of training frames below which a class's weight stops growing


def class_weights(targets: np.ndarray, n_classes: int, alpha: float, floor: float = CLASS_WEIGHT_FLOOR) -> np.ndarray:
    """(n_classes,) loss weights max(f_c, floor) ** -alpha with mean 1 over the training frames; all 1 at alpha 0.

    Example (alpha 0.5): a quality on 4 times as many frames as another gets half its weight.
    """
    f = training_prior(targets, n_classes)
    w = np.maximum(f, floor) ** -alpha
    return w / (f * w).sum()


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict,
          n_classes: int) -> tuple[nn.Module, list[dict]]:
    """Train a causal CRNN with n_classes outputs from scratch with early stopping, as train_v7_2.train.

    cfg["train"]: seed, lr, max_epochs, patience, dropout, excerpt_frames, batch_excerpts, input_lookahead,
    and optionally class_weight_alpha (default 0: no weights) and class_weight_floor.
    """
    t = cfg["train"]
    alpha = t.get("class_weight_alpha", 0)
    weight = None
    if alpha:
        targets = np.concatenate([s.targets for s in train_songs])
        weight = torch.tensor(class_weights(targets, n_classes, alpha, t.get("class_weight_floor", CLASS_WEIGHT_FLOOR)),
                              dtype=torch.float32)
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = ChordCRNN(n_classes, bidirectional=False, dropout=t["dropout"]).to(device())
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])

    history, scores, best = [], [], None
    for epoch in range(1, t["max_epochs"] + 1):
        began = time.time()
        loss = run_epoch(model, opt, train_songs, rng, t, weight)
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
