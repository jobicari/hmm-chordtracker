"""Component 38 (V11): V8's training with a distillation term from frozen teacher networks.

    teacher_log_posteriors(teachers, x)          (B, L, n) log of the teachers' geometric-mean posterior
    distill_loss(student_logits, teacher_logp, mask)   mean KL(teacher || student) over the masked frames
    inside_song(songs, starts, length)           (B, L) True where an excerpt frame lies inside its song
    run_epoch(model, opt, songs, rng, t, teachers)
    train(train_songs, val_sets, cfg, n_classes, teachers, init=None)
    pool_starts(songs, length, rng, n)            (n, 2) random excerpts of unlabelled songs
    train_pool(pool, val_sets, cfg, n_classes, teachers, n_excerpts)   pre-training on the pool, KL only

Loss = chord CE on the real labels (frames with a target, as V8) + kl_weight * KL(teacher || student)
on every frame inside the song, labelled or not (cfg["train"]["kl_weight"], 0 = V8's loss).

Teacher. The geometric mean of the teachers' posteriors, renormalised: log p = mean_i log p_i minus its
log-sum-exp over the classes (as run_v10's step 1 ensemble), at temperature 1. It is computed on the fly,
in eval mode, on the same pitch-shifted excerpt the student sees, so the soft targets are exactly the
ensemble's output for that input. The teachers' forward passes draw no random numbers, so the excerpts,
their order, the shifts and the student's dropout are V8's for the same seed.

Pool (stage A of the plan). Unlabelled recordings as Songs whose targets are all -1. Each epoch draws
n_excerpts excerpts at random from every recording tiled from a random offset, pitch-shifted as V8, and
the loss is the KL term alone. Stage B is train() started from stage A's best model (init).
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from chordtracker import data_v5, data_v6, data_v7_2
from chordtracker.data_v5 import Song
from chordtracker.model_v5 import device
from chordtracker.model_v8 import ChordCRNN
from chordtracker.train_v5 import epochs_since_best
from chordtracker.train_v7_2 import frame_accuracy, frame_loss


@torch.no_grad()
def teacher_log_posteriors(teachers: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """(B, L, n) log posterior of the teachers' geometric mean for the (B, 1, 144, L + 14) strips x."""
    mean = torch.stack([torch.log_softmax(m(x), dim=-1) for m in teachers]).mean(dim=0)
    return mean - torch.logsumexp(mean, dim=-1, keepdim=True)


def distill_loss(student_logits: torch.Tensor, teacher_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the frames where mask is True of KL(teacher || student) = sum_c p_t (log p_t - log p_s)."""
    logq = torch.log_softmax(student_logits, dim=-1)
    kl = (teacher_logp.exp() * (teacher_logp - logq)).sum(dim=-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1)


def inside_song(songs: list[Song], starts: np.ndarray, length: int) -> torch.Tensor:
    """(B, length) True for excerpt frames start .. start + length - 1 that lie inside the song."""
    frames = starts[:, 1:2] + np.arange(length)[None, :]
    n = np.array([len(songs[i].targets) for i in starts[:, 0]])[:, None]
    return torch.from_numpy((frames >= 0) & (frames < n))


def run_epoch(model: nn.Module, opt: torch.optim.Optimizer, songs: list[Song], rng: np.random.Generator,
              t: dict, teachers: list[nn.Module] | None = None) -> tuple[float, float]:
    """train_v7_2.run_epoch plus kl_weight * the distillation term; returns the mean CE and mean KL."""
    model.train()
    dev = next(model.parameters()).device
    length, w = t["excerpt_frames"], t.get("kl_weight", 0)
    starts = data_v6.excerpt_starts(songs, length, rng)
    order = rng.permutation(len(starts))
    shifts = rng.choice(list(data_v5.SHIFTS), size=len(starts))
    ces, kls = [], []
    for first in range(0, len(order), t["batch_excerpts"]):
        pick = order[first:first + t["batch_excerpts"]]
        x, y = data_v7_2.make_batch(songs, starts[pick], shifts[pick], length, t["input_lookahead"])
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        logits = model(x)
        ce = frame_loss(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce
        if w:
            kl = distill_loss(logits, teacher_log_posteriors(teachers, x), inside_song(songs, starts[pick], length).to(dev))
            loss = ce + w * kl
            kls.append(kl.item())
        loss.backward()
        opt.step()
        ces.append(ce.item())
    return float(np.mean(ces)), float(np.mean(kls)) if kls else 0.0


def pool_starts(songs: list[Song], length: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 2) (song index, start frame): every pool song tiled from a random offset, then n tiles at random.

    Unlike data_v6.excerpt_starts no tile is dropped for lack of targets (pool songs have none). If the
    tiles are fewer than n, all of them are returned in a random order.
    """
    rows = [(i, start) for i, song in enumerate(songs)
            for start in range(-int(rng.integers(0, length)), song.cqt.shape[1], length)]
    pick = rng.permutation(len(rows))[:n]
    return np.array(rows, dtype=int).reshape(-1, 2)[pick]


def run_pool_epoch(model: nn.Module, opt: torch.optim.Optimizer, pool: list[Song], rng: np.random.Generator,
                   t: dict, teachers: list[nn.Module], n_excerpts: int) -> float:
    """One pass over n_excerpts random pool excerpts, one random shift each, loss = KL only; the mean KL."""
    model.train()
    dev = next(model.parameters()).device
    length = t["excerpt_frames"]
    starts = pool_starts(pool, length, rng, n_excerpts)
    shifts = rng.choice(list(data_v5.SHIFTS), size=len(starts))
    kls = []
    for first in range(0, len(starts), t["batch_excerpts"]):
        pick = np.arange(first, min(first + t["batch_excerpts"], len(starts)))
        x, _ = data_v7_2.make_batch(pool, starts[pick], shifts[pick], length, t["input_lookahead"])
        x = x.to(dev)
        opt.zero_grad()
        kl = distill_loss(model(x), teacher_log_posteriors(teachers, x), inside_song(pool, starts[pick], length).to(dev))
        kl.backward()
        opt.step()
        kls.append(kl.item())
    return float(np.mean(kls))


def fit(model: nn.Module, epoch, val_sets: dict[str, list[Song]], t: dict) -> tuple[nn.Module, list[dict]]:
    """V8's loop: epoch() -> (ce, kl) each epoch, val frame accuracy, early stopping; the best epoch's model."""
    history, scores, best = [], [], None
    for number in range(1, t["max_epochs"] + 1):
        began = time.time()
        ce, kl = epoch()
        accuracy = {name: frame_accuracy(model, songs, t["input_lookahead"]) for name, songs in val_sets.items()}
        scores.append(float(np.mean(list(accuracy.values()))))
        history.append({"epoch": number, "train_loss": ce, "train_kl": kl, **{f"val_{n}": a for n, a in accuracy.items()},
                        "val_mean": scores[-1], "seconds": time.time() - began})
        print(f"epoch {number:3d}  ce {ce:.3f}  kl {kl:.3f}  "
              + "  ".join(f"{n} {100 * a:.2f}%" for n, a in accuracy.items())
              + f"  {history[-1]['seconds']:.0f} s", flush=True)
        if epochs_since_best(scores) == 0:
            best = copy.deepcopy(model.state_dict())
        elif epochs_since_best(scores) >= t["patience"]:
            break
    model.load_state_dict(best)
    return model, history


def _start(t: dict, n_classes: int, teachers: list[nn.Module] | None, init: dict | None):
    for m in teachers or []:
        m.to(device()).eval().requires_grad_(False)
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = ChordCRNN(n_classes, bidirectional=False, dropout=t["dropout"]).to(device())
    if init is not None:
        model.load_state_dict(init)
    return model, torch.optim.Adam(model.parameters(), lr=t["lr"]), rng


def train(train_songs: list[Song], val_sets: dict[str, list[Song]], cfg: dict, n_classes: int,
          teachers: list[nn.Module] | None = None, init: dict | None = None) -> tuple[nn.Module, list[dict]]:
    """train_v8.train (no class weights) with the distillation term; teachers are frozen, in eval mode.

    init: a state dict to start from (the pool pre-training), else V8's random initialisation.
    """
    t = cfg["train"]
    if t.get("kl_weight", 0) and not teachers:
        raise ValueError("kl_weight > 0 needs teacher models")
    model, opt, rng = _start(t, n_classes, teachers, init)
    return fit(model, lambda: run_epoch(model, opt, train_songs, rng, t, teachers), val_sets, t)


def train_pool(pool: list[Song], val_sets: dict[str, list[Song]], cfg: dict, n_classes: int,
               teachers: list[nn.Module], n_excerpts: int) -> tuple[nn.Module, list[dict]]:
    """Pre-train a student on the unlabelled pool with the KL term alone, n_excerpts random excerpts an
    epoch, early stopping on the val sets' frame accuracy as V8."""
    t = cfg["train"]
    model, opt, rng = _start(t, n_classes, teachers, None)
    return fit(model, lambda: (0.0, run_pool_epoch(model, opt, pool, rng, t, teachers, n_excerpts)), val_sets, t)
