"""Behaviour of the V5 CNN: shapes, size, posteriors, and that it can learn."""
import torch

from chordtracker import model_v5 as m


def test_patches_in_logits_out():
    model = m.ChordCNN()
    assert model(torch.zeros(8, 1, 144, 15)).shape == (8, 24)


def test_about_146k_weights_mostly_in_the_dense_layer():
    model = m.ChordCNN()
    assert 140_000 < m.n_weights(model) < 150_000
    assert m.n_weights(model.classifier) == 64 * 72 * 24 + 24


def test_frame_posteriors_are_one_distribution_per_frame():
    torch.manual_seed(0)
    P = m.frame_posteriors(m.ChordCNN(), torch.rand(5, 1, 144, 15), batch=2)
    assert P.shape == (24, 5)
    assert torch.allclose(P.sum(dim=0), torch.ones(5))


def test_eval_mode_is_deterministic():
    torch.manual_seed(0)
    model, x = m.ChordCNN(), torch.rand(4, 1, 144, 15)
    assert torch.equal(m.frame_posteriors(model, x), m.frame_posteriors(model, x))   # dropout off


def test_it_can_learn_where_a_note_is():
    """Two toy classes: energy on row 30 -> class 0, on row 80 -> class 3. A few steps of Adam separate them."""
    torch.manual_seed(0)
    x = torch.zeros(32, 1, 144, 15)
    y = torch.tensor([0, 3] * 16)
    x[y == 0, 0, 30, :] = 1.0
    x[y == 3, 0, 80, :] = 1.0
    model = m.ChordCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(30):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    assert (m.frame_posteriors(model, x).argmax(dim=0) == y).all()
