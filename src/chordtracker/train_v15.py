"""Component 42 (V15): fine-tuning V14's N model on vocabulary A+ (dim, aug, hdim7), as plain functions.

    extend(model, vocab, prior)                 V14's ChordCRNN with 12 output rows per new quality
    focus_segments(songs, vocab)                every run of new-quality frames in the train songs
    focus_starts(segments, n, length, rng)      n excerpt starts, each holding one such run
    stretched_excerpt(song, start, k, length, lookahead, rate)   input and targets at a tempo change
    make_batch(songs, starts, shifts, rates, length, lookahead, vocab)
    run_epoch(model, opt, songs, focus, rng, aug_rng, t, vocab)  one epoch: replay + focus excerpts
    finetune(model, train_songs, focus, val, t, vocab, evaluate)

Extending the output layer. Every weight of V14 n_seed0 is copied; the classifier gets 12 new rows per
new quality, after N (data_v15: V14's 61 classes keep their indices). Row C:dim starts as a copy of row
C:min, C:aug of C:maj, C:hdim7 of C:min7 (data_v15.Vocab.related: the A chord sharing the root and the
third, and two of the three or four notes), with its bias lowered by log(share of the new quality /
share of the related one) in the natural train labels (about -4). Copying gives the new class the
features of the root it already reads; the lower bias keeps it quiet at the start, so the extended model
before any update decodes as V14 does and the fine-tune grows the new classes from there instead of
first taking half of every minor chord's mass away.

The fine-tune. Each batch holds replay_per_batch excerpts of the normal train data (V14's: the songs
tiled from a random offset, one random shift each, no other augmentation; replay_excerpts of them per
epoch) and focus_per_batch focus excerpts: an excerpt of excerpt_frames (11.9 s) placed at random
around one run of new-quality frames, so the run comes with the chords around it. Focus runs are drawn
quality first (each new quality equally often), then a run of it; the epoch's focus excerpts take the 12
shifts in turn, a time stretch uniform in 1 +/- stretch and a gain uniform in +/- gain_db dB.

Time stretch, on the cached log-CQT (the audio is not read again): output frame t is the song at frame
start + t * rate, its columns interpolated linearly between the two nearest frames, its target the target
of the nearest frame. rate > 1 plays faster. The 15-column window and the input look-ahead stay in
output frames, so the network meets the same chords at a slightly different tempo.

Gain: train_v10.augment's (inside the compression: x' = log(1 + (e^x - 1) g)), on the focus excerpts only.

Choosing the epoch. After every epoch the val posteriors are decoded at V14's demo point (lag 8 p_self
0.98, N weight 0.1, chord -> N rate x0.5, prior power 0.25) and scored (eval_v15): the criterion is the
mean recall of the new qualities, less penalty points per point of 3-set mean sevenths (with N) below
that of epoch 0 (the extended model before any update) minus tolerance:

    criterion = new_recall - penalty * max(0, sevenths_0 - tolerance - sevenths)

model.pt keeps the epoch with the best criterion, model_best_sevenths.pt the one with the best sevenths.
"""
import copy
import time

import numpy as np
import torch

from chordtracker import data_v5, data_v6, model_v14, train_v10
from chordtracker.data_v5 import CENTRE_ROW, PATCH_BINS, Song
from chordtracker.data_v7_2 import CONTEXT
from chordtracker.data_v15 import BASE, N_BASE, Vocab
from chordtracker.train_v7_2 import frame_loss


def extend(model: model_v14.ChordCRNN, vocab: Vocab, prior: np.ndarray) -> model_v14.ChordCRNN:
    """A ChordCRNN with vocab.n_classes outputs: model's weights, new rows from their related rows.

    prior: (n_classes,) natural train prior in model order (for the bias offsets).
    """
    if model.level_input:
        raise ValueError("V15 extends V14's model without the level input")
    out = model_v14.ChordCRNN(vocab.n_classes, level_input=False, dropout=model.classifier[0].p)
    state = model.state_dict()
    old_w, old_b = state["classifier.1.weight"], state["classifier.1.bias"]
    w = torch.zeros(vocab.n_classes, old_w.shape[1])
    b = torch.zeros(vocab.n_classes)
    w[:len(old_w)], b[:len(old_b)] = old_w, old_b
    related = vocab.related()
    for j, q in enumerate(vocab.new):
        rel = BASE.index(related[q])
        rows = slice(N_BASE + 1 + 12 * j, N_BASE + 13 + 12 * j)
        share_new = float(prior[rows].sum())
        share_rel = float(prior[12 * rel:12 * rel + 12].sum())
        w[rows] = old_w[12 * rel:12 * rel + 12]
        b[rows] = old_b[12 * rel:12 * rel + 12] + float(np.log(max(share_new, 1e-6) / share_rel))
    state = {**state, "classifier.1.weight": w, "classifier.1.bias": b}
    out.load_state_dict(state)
    return out


def focus_segments(songs: list[Song], vocab: Vocab) -> list[tuple[int, int, int, int]]:
    """[(song index, first frame, last frame, new quality index)] of every run of one new-quality class."""
    out = []
    for i, song in enumerate(songs):
        y = song.targets
        new = y > N_BASE
        t = 0
        while t < len(y):
            if not new[t]:
                t += 1
                continue
            end = t
            while end + 1 < len(y) and y[end + 1] == y[t]:
                end += 1
            out.append((i, t, end, int((y[t] - N_BASE - 1) // 12)))
            t = end + 1
    return out


def focus_starts(segments: list[tuple[int, int, int, int]], n: int, length: int,
                 rng: np.random.Generator) -> np.ndarray:
    """(n, 2) (song index, start): a quality drawn uniformly, a run of it, an excerpt start that holds the run
    (the run's start, when it is longer than the excerpt, falls anywhere in it)."""
    by_q: dict[int, list] = {}
    for seg in segments:
        by_q.setdefault(seg[3], []).append(seg)
    qualities = sorted(by_q)
    rows = []
    for q in rng.choice(qualities, size=n):
        i, first, last, _ = by_q[q][int(rng.integers(len(by_q[q])))]
        lo, hi = last - length + 1, first
        if lo > hi:
            lo, hi = first, last - length + 1
        rows.append((i, int(rng.integers(lo, hi + 1))))
    return np.array(rows, dtype=int).reshape(-1, 2)


def stretched_excerpt(song: Song, start: int, k: int, length: int, lookahead: int,
                      rate: float) -> tuple[np.ndarray, np.ndarray]:
    """(144, length + 14) input and (length,) model-order targets, k semitones up, at rate (see the docstring).

    rate 1 is data_v7_2.excerpt and data_v6.excerpt_targets (unshifted targets: the caller shifts them).
    """
    top = CENTRE_ROW - 2 * k
    n = song.cqt.shape[1]
    pos = start + (np.arange(length + CONTEXT) - CONTEXT + lookahead) * rate      # source frame of each column
    lo = np.floor(pos).astype(int)
    frac = (pos - lo).astype(np.float32)
    rows = song.cqt[top:top + PATCH_BINS]

    def column(idx):
        inside = (idx >= 0) & (idx < n)
        out = np.zeros((PATCH_BINS, len(idx)), dtype=np.float32)
        out[:, inside] = rows[:, idx[inside]]
        return out

    x = column(lo) * (1 - frac) + column(lo + 1) * frac
    src = np.rint(start + np.arange(length) * rate).astype(int)
    y = np.full(length, -1, dtype=int)
    inside = (src >= 0) & (src < len(song.targets))
    y[inside] = song.targets[src[inside]]
    return x, y


def make_batch(songs: list[Song], starts: np.ndarray, shifts: np.ndarray, rates: np.ndarray, length: int,
               lookahead: int, vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 1, 144, length + 14) inputs and (B, length) model-order targets, each excerpt shifted and stretched."""
    xs, ys = [], []
    for (i, s), k, r in zip(starts, shifts, rates):
        x, y = stretched_excerpt(songs[i], int(s), int(k), length, lookahead, float(r))
        xs.append(x)
        ys.append(vocab.shift_targets(y, int(k)))
    return torch.from_numpy(np.stack(xs)[:, None]), torch.from_numpy(np.stack(ys)).long()


def run_epoch(model, opt, songs: list[Song], focus: list, rng: np.random.Generator, aug_rng: np.random.Generator,
              t: dict, vocab: Vocab) -> tuple[float, np.ndarray]:
    """One epoch of replay + focus batches; (mean loss, (n_classes,) count of the model-order targets trained)."""
    model.train()
    dev = next(model.parameters()).device
    length, la = t["excerpt_frames"], t["input_lookahead"]
    n_rep, n_foc = t["replay_per_batch"], t["focus_per_batch"]
    replay = data_v6.excerpt_starts(songs, length, rng)
    replay = replay[rng.permutation(len(replay))[:t["replay_excerpts"]]]
    n_batches = int(np.ceil(len(replay) / n_rep))
    foc = focus_starts(focus, n_batches * n_foc, length, rng)
    foc_shifts = rng.permutation(np.resize(np.array(list(data_v5.SHIFTS)), len(foc)))
    foc_rates = 1.0 + rng.uniform(-1, 1, len(foc)) * t["stretch"]
    rep_shifts = rng.choice(list(data_v5.SHIFTS), size=len(replay))
    losses, counts = [], np.zeros(vocab.n_classes)
    for b in range(n_batches):
        r = slice(b * n_rep, (b + 1) * n_rep)
        f = slice(b * n_foc, (b + 1) * n_foc)
        xr, yr = make_batch(songs, replay[r], rep_shifts[r], np.ones(len(replay[r])), length, la, vocab)
        xf, yf = make_batch(songs, foc[f], foc_shifts[f], foc_rates[f], length, la, vocab)
        if t.get("gain_db", 0):
            xf = torch.from_numpy(train_v10.augment(xf.numpy(), aug_rng, {"gain_db": t["gain_db"]}))
        x, y = torch.cat([xr, xf]), torch.cat([yr, yf])
        counts += np.bincount(y[y >= 0].numpy(), minlength=vocab.n_classes)
        opt.zero_grad()
        logits = model(x.to(dev))
        loss = frame_loss(logits.reshape(-1, logits.shape[-1]), y.to(dev).reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses)), counts


def criterion(sevenths: float, recall: float, sevenths_0: float, t: dict) -> float:
    """new_recall - penalty * max(0, sevenths_0 - tolerance - sevenths), all in %."""
    return recall - t["penalty"] * max(0.0, sevenths_0 - t["tolerance"] - sevenths)


def optimiser(model, t: dict) -> torch.optim.Optimizer:
    """Adam at lr, the output layer at head_lr (its new rows have the furthest to go)."""
    head = list(model.classifier.parameters())
    ids = {id(p) for p in head}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.Adam([{"params": rest, "lr": t["lr"]}, {"params": head, "lr": t["head_lr"]}])


def finetune(model, train_songs: list[Song], focus: list, t: dict, vocab: Vocab, evaluate,
             log=print) -> tuple[object, object, list[dict], np.ndarray]:
    """(best-criterion model, best-sevenths model, history, summed target counts of the epochs trained).

    evaluate(model, mix_counts) -> {'sevenths': %, 'recall': %, ...} at the demo point; mix_counts are the
    (n_classes,) target counts trained so far (None at epoch 0), for the decoder's prior.
    """
    torch.manual_seed(t["seed"])
    rng = np.random.default_rng(t["seed"])
    aug_rng = np.random.default_rng(t["seed"] + train_v10.AUGMENT_SEED_OFFSET)
    opt = optimiser(model, t)
    began = time.time()
    e0 = evaluate(model, None)
    history = [{"epoch": 0, "train_loss": None, **e0, "criterion": criterion(e0["sevenths"], e0["recall"],
                                                                              e0["sevenths"], t),
                "seconds": time.time() - began}]
    log(f"epoch   0  " + "  ".join(f"{k} {v:.2f}" for k, v in e0.items() if isinstance(v, float)), flush=True)
    best, best_sev = copy.deepcopy(model.state_dict()), copy.deepcopy(model.state_dict())
    counts = np.zeros(vocab.n_classes)
    for epoch in range(1, t["max_epochs"] + 1):
        began = time.time()
        loss, c = run_epoch(model, opt, train_songs, focus, rng, aug_rng, t, vocab)
        counts += c
        e = evaluate(model, counts)
        crit = criterion(e["sevenths"], e["recall"], e0["sevenths"], t)
        history.append({"epoch": epoch, "train_loss": loss, **e, "criterion": crit, "seconds": time.time() - began})
        log(f"epoch {epoch:3d}  loss {loss:.3f}  criterion {crit:.2f}  "
            + "  ".join(f"{k} {v:.2f}" for k, v in e.items() if isinstance(v, float))
            + f"  {history[-1]['seconds']:.0f} s", flush=True)
        crits = [h["criterion"] for h in history]
        if int(np.argmax(crits)) == len(crits) - 1:
            best = copy.deepcopy(model.state_dict())
        if int(np.argmax([h["sevenths"] for h in history])) == len(history) - 1:
            best_sev = copy.deepcopy(model.state_dict())
        if len(crits) - 1 - int(np.argmax(crits)) >= t["patience"]:
            break
    best_model, sev_model = copy.deepcopy(model), copy.deepcopy(model)
    best_model.load_state_dict(best)
    sev_model.load_state_dict(best_sev)
    return best_model, sev_model, history, counts
