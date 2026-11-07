"""Behaviour of the class-weighted loss: alpha 0 is the plain loss, rarer classes weigh more, mean weight 1."""
import numpy as np
import pytest
import torch

from chordtracker import train_v7_2, train_v8
from chordtracker.transitions_v8 import training_prior

# A: maj on 6 frames, min on 3, 7 on 1, maj7 and min7 never
TARGETS = np.array([0, 2, 7, 7, 9, 5, 21, 14, 16, 31, -1, -1])


def test_alpha_0_gives_every_class_weight_1():
    assert np.array_equal(train_v8.class_weights(TARGETS, 60, 0.0), np.ones(60))


def test_alpha_0_weights_give_the_unweighted_loss():
    torch.manual_seed(0)
    logits = torch.randn(len(TARGETS), 60)
    y = torch.from_numpy(TARGETS)
    plain = train_v7_2.frame_loss(logits, y)
    weighted = train_v7_2.frame_loss(logits, y, torch.tensor(train_v8.class_weights(TARGETS, 60, 0.0)).float())
    assert torch.allclose(plain, weighted)
    assert torch.allclose(plain, torch.nn.functional.cross_entropy(logits, y, ignore_index=-1))


def test_a_rarer_class_gets_a_bigger_weight_pooled_over_its_roots():
    w = train_v8.class_weights(TARGETS, 60, 0.5)
    assert np.allclose(w[:12], w[0]) and np.allclose(w[24:36], w[24])         # one weight per quality
    assert w[24] > w[12] > w[0]                                                 # 7 rarer than min rarer than maj
    assert np.isclose(w[12] / w[0], np.sqrt(2))                                 # half the frames, sqrt(2) the weight
    assert w[36] == w[48] > w[24]                                               # absent: the floor, the largest


def test_the_floor_bounds_the_weight_of_an_absent_class():
    low, high = train_v8.class_weights(TARGETS, 60, 0.5, floor=1e-6), train_v8.class_weights(TARGETS, 60, 0.5, floor=1e-2)
    assert np.isfinite(low).all() and low[36] > high[36]


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
def test_the_mean_weight_over_the_training_frames_is_1(alpha):
    w = train_v8.class_weights(TARGETS, 60, alpha)
    assert np.isclose((training_prior(TARGETS, 60) * w).sum(), 1)


def test_weighted_loss_scales_each_frame_by_its_class_weight():
    torch.manual_seed(1)
    logits = torch.randn(len(TARGETS), 60)
    y = torch.from_numpy(TARGETS)
    weight = torch.tensor(train_v8.class_weights(TARGETS, 60, 0.5)).float()
    per_frame = torch.nn.functional.cross_entropy(logits, y, ignore_index=-1, reduction="none")
    keep = y >= 0
    expected = (per_frame[keep] * weight[y[keep]]).sum() / keep.sum()
    assert torch.allclose(train_v7_2.frame_loss(logits, y, weight), expected)
