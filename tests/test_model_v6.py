"""Behaviour of the V6 CRNN: shapes, size, V5's front end, which frames a frame can see, and that it can learn."""
import torch

from chordtracker import data_v6, model_v5
from chordtracker import model_v6 as m


def test_a_strip_in_one_score_vector_per_frame_out():
    for bidirectional in (True, False):
        model = m.ChordCRNN(bidirectional=bidirectional)
        assert model(torch.zeros(3, 1, 144, 20 + 14)).shape == (3, 20, 24)


def test_no_bigger_than_v5():
    v5 = model_v5.n_weights(model_v5.ChordCNN())
    assert model_v5.n_weights(m.ChordCRNN(bidirectional=False)) < model_v5.n_weights(m.ChordCRNN()) < v5


def test_the_window_of_frame_t_is_v5s_patch():
    """Window t of an excerpt strip is the 15 columns data_v5.patch cuts for that frame."""
    x = torch.rand(2, 1, 144, 10 + 14)
    w = m.windows(x)
    assert w.shape == (20, 1, 144, 15)
    assert torch.equal(w[13], x[1, :, :, 3:18])                # excerpt 1, frame 3


def test_the_front_end_is_v5s_layers():
    shapes = lambda net: [tuple(p.shape) for p in net.parameters()]
    assert shapes(m.ChordCRNN().front) == shapes(model_v5.ChordCNN().features)


def scores_after_changing_the_end(model, t):
    """Logits of frames 0..t before and after the input of every frame after t is replaced."""
    torch.manual_seed(1)
    x = torch.rand(1, 1, 144, 30 + 14)
    y = x.clone()
    y[..., t + 1 + data_v6.MARGIN:] = torch.rand_like(y[..., t + 1 + data_v6.MARGIN:])
    model.eval()
    with torch.no_grad():
        return model(x)[0, :t + 1 - data_v6.MARGIN], model(y)[0, :t + 1 - data_v6.MARGIN]


def test_a_causal_model_does_not_look_ahead_beyond_its_window():
    """Frames whose own 15-frame window ends before the change keep their scores."""
    torch.manual_seed(0)
    before, after = scores_after_changing_the_end(m.ChordCRNN(bidirectional=False), t=20)
    assert torch.allclose(before, after)


def test_a_bidirectional_model_does():
    torch.manual_seed(0)
    before, after = scores_after_changing_the_end(m.ChordCRNN(bidirectional=True), t=20)
    assert not torch.allclose(before, after)


def test_frame_posteriors_match_the_forward_pass_and_sum_to_one():
    torch.manual_seed(0)
    model, x = m.ChordCRNN(), torch.rand(1, 1, 144, 25 + 14)
    P = m.frame_posteriors(model, x, chunk=7)
    assert P.shape == (24, 25)
    assert torch.allclose(P.sum(dim=0), torch.ones(25))
    with torch.no_grad():
        assert torch.allclose(P, torch.softmax(model(x), dim=2)[0].T, atol=1e-6)


def test_it_can_learn_a_chord_that_only_its_neighbours_reveal():
    """A frame with no input is class 1 after a note and class 2 before one: only the GRU can tell."""
    torch.manual_seed(0)
    margin = data_v6.MARGIN
    x = torch.zeros(4, 1, 144, 12 + 14)
    y = torch.zeros(4, 12, dtype=torch.long)
    x[:2, 0, 40, margin:margin + 4] = 1.0                     # excerpts 0, 1: note first, then silence
    y[:2, 4:] = 1
    x[2:, 0, 40, margin + 8:margin + 12] = 1.0                # excerpts 2, 3: silence first, then the note
    y[2:, :8] = 2
    model = m.ChordCRNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(100):
        model.train()
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x).reshape(-1, 24), y.reshape(-1)).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        assert (model(x).argmax(dim=2) == y).all()
