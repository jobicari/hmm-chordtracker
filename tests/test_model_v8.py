"""Behaviour of the V8 CRNN: n_classes outputs, model_v6 exactly at 24, the weight count of the plan."""
import torch

from chordtracker import model_v6, model_v8


def test_outputs_one_score_per_class():
    x = torch.zeros(2, 1, 144, 30 + 14)
    for n in (60, 84, 96):
        assert model_v8.ChordCRNN(n)(x).shape == (2, 30, n)


def test_at_24_classes_it_is_model_v6():
    torch.manual_seed(0)
    v6 = model_v6.ChordCRNN(bidirectional=False)
    torch.manual_seed(0)
    v8 = model_v8.ChordCRNN(24)
    assert v6.state_dict().keys() == v8.state_dict().keys()
    assert all(torch.equal(a, b) for a, b in zip(v6.state_dict().values(), v8.state_dict().values()))


def test_weight_counts():
    assert model_v8.n_weights(model_v8.ChordCRNN(24)) == 99_152
    assert model_v8.n_weights(model_v8.ChordCRNN(60)) == 101_492


def test_frame_posteriors_have_one_row_per_class():
    strip = torch.zeros(1, 1, 144, 50 + 14)
    post = model_v6.frame_posteriors(model_v8.ChordCRNN(60), strip)
    assert post.shape == (60, 50)
    torch.testing.assert_close(post.sum(dim=0), torch.ones(50))
