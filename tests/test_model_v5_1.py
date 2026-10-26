"""Behaviour of the V5.1 CNN: 25 outputs, V5's blocks, and that it can learn N."""
import torch

from chordtracker import model_v5, model_v5_1 as m


def test_patches_in_25_logits_out():
    assert m.ChordCNNWithN()(torch.zeros(8, 1, 144, 15)).shape == (8, 25)


def test_one_more_dense_row_than_v5():
    new, old = m.ChordCNNWithN(), model_v5.ChordCNN()
    assert model_v5.n_weights(new) - model_v5.n_weights(old) == 64 * 72 + 1
    assert model_v5.n_weights(new.features) == model_v5.n_weights(old.features)


def test_frame_posteriors_have_25_rows():
    torch.manual_seed(0)
    P = model_v5.frame_posteriors(m.ChordCNNWithN(), torch.rand(5, 1, 144, 15), batch=2)
    assert P.shape == (25, 5)
    assert torch.allclose(P.sum(dim=0), torch.ones(5))


def test_it_can_tell_silence_from_a_note():
    """Toy: an empty patch -> N (24), energy on row 30 -> class 0. Adam separates them.

    100 steps, not V5's 30: eval mode uses batch norm's running statistics, which move 10% per step,
    and an all-zero class shifts them far enough that after 30 steps they still lag behind training.
    """
    torch.manual_seed(0)
    x = torch.zeros(32, 1, 144, 15)
    y = torch.tensor([24, 0] * 16)
    x[y == 0, 0, 30, :] = 1.0
    model = m.ChordCNNWithN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        model.train()
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    assert (model_v5.frame_posteriors(model, x).argmax(dim=0) == y).all()
