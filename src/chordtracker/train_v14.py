"""Component 39 (V14): training V8's causal CRNN with an N class, and optionally the input level, as plain functions.

train_v8's loop (the V8 recipe: excerpts, one random shift each, Adam, early stopping on the mean val frame
accuracy) with three differences:

    targets      vocabulary A plus N (data_v8.target with with_n), so N frames are trained, not left out
    shift        N keeps its index under the pitch shift (data_v7_2.make_batch with n_chords)
    level        with cfg["train"]["level_input"], the GRU also reads the frame level (model_v14, features_v14)

    LevelSong                                         data_v5.Song plus its (n_frames,) scaled level, or None
    excerpt_level(song, start, length, lookahead)     (length,) the level the GRU reads at each frame
    make_batch(songs, starts, shifts, length, lookahead, n_chords, level_input)
    song_posteriors(model, song, lookahead)           (n_classes, n_frames), whole song, unshifted
    frame_accuracy(model, songs, lookahead)
    train(train_songs, val_sets, cfg, n_classes, n_chords)

The level is not shifted: a pitch shift does not change how loud a frame is.
"""
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from chordtracker import data_v5, data_v6, data_v7_2, model_v14
from chordtracker.data_v5 import Song
from chordtracker.model_v5 import device
from chordtracker.train_v5 import epochs_since_best
from chordtracker.train_v6 import save  # noqa: F401  (same weights format as V6)
from chordtracker.train_v7_2 import frame_loss


@dataclass(frozen=True)
class LevelSong(Song):
    """A Song with the scaled dBFS level of every frame (features_v14), or None when the model does not read it."""
    level: np.ndarray | None = None


def excerpt_level(song: LevelSong, start: int, length: int, lookahead: int) -> np.ndarray:
    """(length,) level of frames start + a .. start + length - 1 + a; silence (-1) outside the song."""
    out = np.full(length, model_v14.SILENCE, dtype=np.float32)
    first = start + lookahead
    lo, hi = max(first, 0), min(first + length, len(song.level))
    if lo < hi:
        out[lo - first:hi - first] = song.level[lo:hi]
    return out


def make_batch(songs: list[LevelSong], starts: np.ndarray, shifts: np.ndarray, length: int, lookahead: int,
               n_chords: int, level_input: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """data_v7_2.make_batch with N kept under the shift, plus the (B, length) level strip (None without level)."""
    x, y = data_v7_2.make_batch(songs, starts, shifts, length, lookahead, n_chords)
    if not level_input:
        return x, y, None
    lv = np.stack([excerpt_level(songs[i], s, length, lookahead) for i, s in starts])
    return x, y, torch.from_numpy(lv)


def run_epoch(model: model_v14.ChordCRNN, opt: torch.optim.Optimizer, songs: list[LevelSong],
              rng: np.random.Generator, t: dict, n_chords: int) -> float:
    """One pass over this epoch's excerpts in a random order, one random shift each; returns the mean loss."""
    model.train()
    dev = next(model.parameters()).device
    length = t["excerpt_frames"]
    starts = data_v6.excerpt_starts(songs, length, rng)
    order = rng.permutation(len(starts))
    shifts = rng.choice(list(data_v5.SHIFTS), size=len(starts))
    losses = []
    for first in range(0, len(order), t["batch_excerpts"]):
        pick = order[first:first + t["batch_excerpts"]]
        x, y, lv = make_batch(songs, starts[pick], shifts[pick], length, t["input_lookahead"], n_chords,
                              model.level_input)
        opt.zero_grad()
        logits = model(x.to(dev), None if lv is None else lv.to(dev))
        loss = frame_loss(logits.reshape(-1, logits.shape[-1]), y.to(dev).reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def song_posteriors(model: model_v14.ChordCRNN, song: LevelSong, lookahead: int) -> np.ndarray:
    """(n_classes, n_frames) frame posteriors of a whole song read as one excerpt, unshifted, eval mode."""
    n = song.cqt.shape[1]
    strip = torch.from_numpy(data_v7_2.excerpt(song, 0, 0, n, lookahead))[None, None]
    level = torch.from_numpy(excerpt_level(song, 0, n, lookahead))[None] if model.level_input else None
    return model_v14.frame_posteriors(model, strip, level).numpy()


def frame_accuracy(model: model_v14.ChordCRNN, songs: list[LevelSong], lookahead: int) -> float:
    """Share of frames with a target (N included) whose most likely class is the target, pooled over the songs."""
    right = total = 0
    for song in songs:
        keep = song.targets >= 0
        right += int((song_posteriors(model, song, lookahead).argmax(axis=0)[keep] == song.targets[keep]).sum())
        total += int(keep.sum())
    return right / total


def train(train_songs: list[LevelSong], val_sets: dict[str, list[LevelSong]], cfg: dict, n_classes: int,
          n_chords: int) -> tuple[model_v14.ChordCRNN, list[dict]]:
    """Train from scratch with early stopping, as train_v8.train; the best epoch's model and one row per epoch.

    cfg["train"]: seed, lr, max_epochs, patience, dropout, excerpt_frames, batch_excerpts, input_lookahead,
    level_input (default False).
    """
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    model = model_v14.ChordCRNN(n_classes, level_input=t.get("level_input", False), dropout=t["dropout"]).to(device())
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])

    history, scores, best = [], [], None
    for epoch in range(1, t["max_epochs"] + 1):
        began = time.time()
        loss = run_epoch(model, opt, train_songs, rng, t, n_chords)
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
