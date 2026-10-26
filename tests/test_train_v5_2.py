"""Tests for train_v5_2: the optimizer choice and what weight decay touches."""
import numpy as np
import torch

from chordtracker import train_v5_2
from chordtracker.model_v5 import ChordCNN


def test_no_decay_is_v5s_adam():
    opt = train_v5_2.optimizer(ChordCNN(), {"lr": 1e-3, "weight_decay": 0.0})
    assert type(opt) is torch.optim.Adam
    assert type(train_v5_2.optimizer(ChordCNN(), {"lr": 1e-3})) is torch.optim.Adam


def test_decay_skips_biases_and_batch_norm():
    model = ChordCNN()
    opt = train_v5_2.optimizer(model, {"lr": 1e-3, "weight_decay": 1e-2})
    assert type(opt) is torch.optim.AdamW
    decayed, kept = opt.param_groups
    assert decayed["weight_decay"] == 1e-2 and kept["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in decayed["params"]) and all(p.ndim <= 1 for p in kept["params"])
    assert len(decayed["params"]) + len(kept["params"]) == len(list(model.parameters()))


def test_decay_alone_shrinks_the_weights():
    """With a zero gradient, AdamW still pulls the decayed weights towards zero; biases stay put."""
    torch.manual_seed(0)
    model = ChordCNN()
    opt = train_v5_2.optimizer(model, {"lr": 1e-2, "weight_decay": 1e-1})
    before = train_v5_2.weight_norm(model)
    biases = [p.detach().clone() for p in model.parameters() if p.ndim <= 1]
    for _ in range(10):
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
    assert train_v5_2.weight_norm(model) < before
    assert all(torch.equal(a, p) for a, p in zip(biases, [p for p in model.parameters() if p.ndim <= 1]))


def test_epoch_examples_mix():
    rng = np.random.default_rng(0)
    real = np.array([[0, t] for t in range(10)])
    extra = np.array([[1, t] for t in range(100)])
    assert train_v5_2.epoch_examples(real, extra, 0, rng) is real                  # V5: no AAM, no draw
    mixed = train_v5_2.epoch_examples(real, extra, 1, rng)
    assert len(mixed) == 20 and (mixed[:10] == real).all()
    assert len({tuple(e) for e in mixed[10:]}) == 10                                # without replacement
    assert len(train_v5_2.epoch_examples(real, extra, 2, rng)) == 30
    assert len(train_v5_2.epoch_examples(real, extra, -1, rng)) == 110              # all of AAM
    assert len(train_v5_2.epoch_examples(real, extra[:5], 1, rng)) == 15            # capped at what exists
    first = train_v5_2.epoch_examples(real, extra, 1, np.random.default_rng(1))[10:]
    second = train_v5_2.epoch_examples(real, extra, 1, np.random.default_rng(2))[10:]
    assert not (first == second).all()                                              # a new draw each epoch
