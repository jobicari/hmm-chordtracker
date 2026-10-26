"""Component 16 (V5.2): V5's training with weight decay (AdamW).

    optimizer(model, t)                       Adam (weight_decay 0, as V5) or AdamW with decay on the weights
    epoch_examples(real, extra, ratio, rng)   this epoch's examples: all real frames + a fresh draw of AAM frames
    train(train_songs, val_sets, cfg, aam_songs)   train_v5.train with both -> best model + history

Weight decay shrinks every weight a little at each step (decoupled from the gradient, AdamW), so the
network can only keep large weights the data keeps asking for. It is applied to the conv kernels and
the dense matrix only, not to the biases or the batch-norm scales: those set offsets and gains, and
shrinking them towards zero does not make the model any simpler.

AAM (step 2 on): each epoch sees every real training frame once, plus ratio x as many AAM frames
drawn at random without replacement, a new draw every epoch. With 1,000 AAM songs (~1.7M frames) and
163k real frames, a 1:1 epoch sees about a tenth of AAM, so the network keeps meeting new songs.
The real songs keep at least half the weight of every epoch, and early stopping still looks at the
real val songs only. ratio -1 means every AAM frame, every epoch.

Everything else is train_v5: the epochs, random shifts, batch, early stopping on the mean val frame
accuracy, and the saved file (train_v5.save / train_v5.load read it unchanged).
"""
import copy

import numpy as np
import torch
from torch import nn

from chordtracker.data_v5 import Song
from chordtracker.model_v5 import ChordCNN, device
from chordtracker.train_v5 import epochs_since_best, frame_accuracy, run_epoch, training_frames


def optimizer(model: nn.Module, t: dict) -> torch.optim.Optimizer:
    """V5's Adam when t["weight_decay"] is 0 or missing; otherwise AdamW, decay on tensors of 2+ dims."""
    decay = t.get("weight_decay", 0.0)
    if not decay:
        return torch.optim.Adam(model.parameters(), lr=t["lr"])
    weights = [p for p in model.parameters() if p.ndim > 1]
    others = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.AdamW([{"params": weights, "weight_decay": decay},
                              {"params": others, "weight_decay": 0.0}], lr=t["lr"])


def epoch_examples(real: np.ndarray, extra: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """All real examples plus round(ratio x len(real)) extra ones drawn without replacement (-1: all)."""
    if ratio == 0 or len(extra) == 0:
        return real
    if ratio < 0:
        return np.concatenate([real, extra])
    n = min(int(round(ratio * len(real))), len(extra))
    return np.concatenate([real, extra[rng.choice(len(extra), size=n, replace=False)]])


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict,
          aam_songs: list[Song] = ()) -> tuple[nn.Module, list[dict]]:
    """train_v5.train with optimizer(model, cfg["train"]) and AAM frames mixed into every epoch.

    cfg["train"]: seed, lr, batch, max_epochs, patience, dropout, weight_decay, aam_ratio (0 = none).
    History rows also carry the weight norm and the number of examples seen.
    """
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = ChordCNN(dropout=t["dropout"]).to(device())
    opt = optimizer(model, t)
    songs = list(train_songs) + list(aam_songs)
    real = training_frames(list(train_songs))
    extra = training_frames(list(aam_songs)) + [len(train_songs), 0] if aam_songs else np.empty((0, 2), int)

    history, scores, best = [], [], None
    for epoch in range(1, t["max_epochs"] + 1):
        examples = epoch_examples(real, extra, t.get("aam_ratio", 0), rng)
        loss = run_epoch(model, opt, songs, examples, rng, t["batch"])
        accuracy = {name: frame_accuracy(model, songs) for name, songs in val_sets.items()}
        scores.append(float(np.mean(list(accuracy.values()))))
        history.append({"epoch": epoch, "train_loss": loss, **{f"val_{n}": a for n, a in accuracy.items()},
                        "val_mean": scores[-1], "weight_norm": weight_norm(model), "examples": len(examples)})
        print(f"epoch {epoch:3d}  loss {loss:.3f}  |w| {history[-1]['weight_norm']:.1f}  "
              + "  ".join(f"{n} {100 * a:.2f}%" for n, a in accuracy.items()), flush=True)
        if epochs_since_best(scores) == 0:
            best = copy.deepcopy(model.state_dict())
        elif epochs_since_best(scores) >= t["patience"]:
            break
    model.load_state_dict(best)
    return model, history


def weight_norm(model: nn.Module) -> float:
    """L2 norm of the decayed tensors (conv kernels and dense matrix): what weight decay pulls down."""
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters() if p.ndim > 1)))
