"""train_v11: the teachers' geometric mean, the distillation loss, the in-song mask and one training epoch."""
import numpy as np
import pytest
import torch

from chordtracker import model_v6, train_v7_2, train_v11
from chordtracker.data_v5 import Song
from chordtracker.model_v8 import ChordCRNN


class Fixed(torch.nn.Module):
    """A teacher with fixed (B, L, n) logits, whatever the input."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        return self.logits


def test_one_teacher_is_its_own_posterior_and_two_are_the_renormalised_geometric_mean():
    a, b = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
    one = train_v11.teacher_log_posteriors([Fixed(a)], None)
    assert torch.allclose(one, torch.log_softmax(a, dim=-1), atol=1e-6)
    two = train_v11.teacher_log_posteriors([Fixed(a), Fixed(b)], None).exp()
    g = torch.sqrt(torch.softmax(a, -1) * torch.softmax(b, -1))
    assert torch.allclose(two, g / g.sum(-1, keepdim=True), atol=1e-6)
    assert torch.allclose(two.sum(-1), torch.ones(2, 3), atol=1e-6)


def test_distill_loss_is_zero_on_the_teacher_positive_elsewhere_and_masked():
    logits = torch.randn(2, 4, 6)
    mask = torch.ones(2, 4, dtype=torch.bool)
    assert train_v11.distill_loss(logits, torch.log_softmax(logits, -1), mask).item() == pytest.approx(0, abs=1e-6)
    other = torch.log_softmax(torch.randn(2, 4, 6), -1)
    assert train_v11.distill_loss(logits, other, mask).item() > 0
    mask[1] = False                                           # the masked excerpt's frames do not count
    wrong = other.clone()
    wrong[1] = torch.log_softmax(torch.randn(4, 6) * 10, -1)
    assert torch.isclose(train_v11.distill_loss(logits, other, mask), train_v11.distill_loss(logits, wrong, mask))


def test_inside_song_marks_the_frames_of_the_song():
    songs = [Song("a", np.zeros((168, 5), np.float32), np.zeros(5, int)),
             Song("b", np.zeros((168, 3), np.float32), np.full(3, -1))]
    mask = train_v11.inside_song(songs, np.array([[0, -2], [1, 1]]), 4)
    assert mask.tolist() == [[False, False, True, True], [True, True, False, False]]


def tiny_songs(n: int, rng) -> list[Song]:
    return [Song(f"s{i}", rng.uniform(0, 2, (168, 40)).astype(np.float32), rng.integers(-1, 60, 40)) for i in range(n)]


T = {"seed": 0, "lr": 1e-3, "max_epochs": 1, "patience": 5, "dropout": 0.5, "excerpt_frames": 16,
     "batch_excerpts": 2, "input_lookahead": 2}


def test_without_the_kl_term_an_epoch_is_v8s():
    songs = tiny_songs(2, np.random.default_rng(0))
    out = []
    for epoch in (train_v7_2.run_epoch, train_v11.run_epoch):
        torch.manual_seed(0)
        model = ChordCRNN(60, bidirectional=False)
        loss = epoch(model, torch.optim.Adam(model.parameters(), lr=1e-3), songs, np.random.default_rng(0), T)
        out.append((loss if np.isscalar(loss) else loss[0], model.state_dict()))
    assert out[0][0] == pytest.approx(out[1][0])
    assert all(torch.equal(v, out[1][1][k]) for k, v in out[0][1].items())


def test_one_distilled_epoch_trains_and_leaves_the_teachers_alone():
    songs = tiny_songs(2, np.random.default_rng(0))
    torch.manual_seed(1)
    teacher = ChordCRNN(60, bidirectional=False)
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    model, history = train_v11.train(songs, {"a": songs[:1]}, {"train": {**T, "kl_weight": 1.0}}, 60, [teacher])
    assert len(history) == 1 and history[0]["train_kl"] > 0 and np.isfinite(history[0]["train_loss"])
    assert all(torch.equal(v, teacher.state_dict()[k].cpu()) for k, v in before.items())
    assert model_v6.frame_posteriors(model, torch.rand(1, 1, 144, 30 + 14)).shape == (60, 30)


def test_kl_weight_needs_teachers():
    with pytest.raises(ValueError):
        train_v11.train([], {}, {"train": {**T, "kl_weight": 1.0}}, 60)
