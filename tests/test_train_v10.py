"""train_v10: bass targets, the heads model, augmentation and one training epoch with every option."""
import numpy as np
import pytest
import torch

from chordtracker import data_v8, frame_score, model_v6, train_v10
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5 import Song
from chordtracker.model_v10 import ChordCRNNHeads
from chordtracker.model_v8 import ChordCRNN


def test_bass_targets_docstring_examples():
    labels = ["C:maj", "C:maj/3", "A:min7/b7", "D:dim", "N"]
    ann = ChordAnnotation(np.array([[i, i + 1.0] for i in range(5)]), labels)
    assert train_v10.bass_targets(ann, np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])).tolist() == [0, 4, 7, 2, -1, -1]


def test_excerpt_bass_shifts_with_the_chord_and_pads():
    bass = np.array([0, 11, -1, 4])
    assert train_v10.excerpt_bass(bass, -1, 2, 6).tolist() == [-1, 2, 1, -1, 6, -1]


def test_heads_model_keeps_the_chord_output():
    torch.manual_seed(0)
    plain = ChordCRNN(60, bidirectional=False)
    torch.manual_seed(0)
    heads = ChordCRNNHeads(60, 5)
    shared = {k: v for k, v in heads.state_dict().items() if not k.startswith("aux.")}
    assert all(torch.equal(v, plain.state_dict()[k]) for k, v in shared.items())     # same start as V8
    heads.eval()
    x = torch.rand(2, 1, 144, 20 + 14)
    chord, root, quality, bass = heads.heads(x)
    assert chord.shape == (2, 20, 60) and root.shape == (2, 20, 12) and quality.shape == (2, 20, 5)
    assert bass.shape == (2, 20, 12) and torch.allclose(heads(x), chord)


def test_augment_is_the_identity_at_zero_and_a_gain_scales_c_over_level():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2, (3, 1, 144, 10)).astype(np.float32)
    assert np.allclose(train_v10.augment(x, rng, {"gain_db": 0, "tilt_db": 0, "noise": 0}), x, atol=1e-6)
    out = train_v10.augment(x, np.random.default_rng(1), {"gain_db": 6})
    ratio = np.expm1(out) / np.expm1(x)                       # one factor per excerpt, within +/- 6 dB
    assert np.allclose(ratio, ratio[:, :, :1, :1], rtol=1e-4)
    assert np.all((ratio >= 10 ** (-6 / 20) - 1e-4) & (ratio <= 10 ** (6 / 20) + 1e-4))


def test_noise_leaves_the_padding_silent():
    x = np.zeros((1, 1, 144, 6), dtype=np.float32)
    x[..., 3:] = 1.0
    out = train_v10.augment(x, np.random.default_rng(0), {"noise": 0.1})
    assert np.all(out[..., :3] == 0) and np.all(out >= 0) and not np.allclose(out[..., 3:], 1.0)


def tiny_songs(n: int, rng) -> list[Song]:
    return [Song(f"s{i}", rng.uniform(0, 2, (168, 40)).astype(np.float32), rng.integers(-1, 60, 40)) for i in range(n)]


@pytest.mark.parametrize("options", [{}, {"early_stop": "sevenths", "aux_weight": 0.5,
                                          "augment": {"gain_db": 3, "tilt_db": 6, "noise": 0.05}}])
def test_one_epoch_trains_with_and_without_the_options(options):
    rng = np.random.default_rng(0)
    songs = tiny_songs(2, rng)
    t = {"seed": 0, "lr": 1e-3, "max_epochs": 1, "patience": 5, "dropout": 0.5, "excerpt_frames": 16,
         "batch_excerpts": 2, "input_lookahead": 2, **options}
    period = 2048 / 22050
    ann = ChordAnnotation(np.array([[0.0, 2.0], [2.0, 40 * period]]), ["C:maj", "G:7"])
    labels = data_v8.class_names("A")
    scored = {"a": [(songs[0], *frame_score.tables(ann, labels, 40, period, "sevenths"))]}
    bass = [rng.integers(-1, 12, 40) for _ in songs]
    model, history = train_v10.train(songs, {"a": songs[:1]}, {"train": t}, 60, bass, scored)
    assert len(history) == 1 and np.isfinite(history[0]["train_loss"]) and "val_sevenths_a" in history[0]
    assert isinstance(model, ChordCRNNHeads if options else ChordCRNN)
    assert model_v6.frame_posteriors(model, torch.rand(1, 1, 144, 30 + 14)).shape == (60, 30)


def test_sevenths_stopping_needs_the_tables():
    with pytest.raises(ValueError):
        train_v10.train([], {}, {"train": {"seed": 0, "early_stop": "sevenths"}}, 60)
