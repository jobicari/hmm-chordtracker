"""Behaviour of the frame-by-frame CRNN: it gives the whole-song posteriors, and only for a causal model."""
import numpy as np
import pytest
import torch

from chordtracker import model_v6
from chordtracker.streaming_crnn import StreamingCRNN


def posteriors_both_ways(n_frames):
    torch.manual_seed(0)
    model = model_v6.ChordCRNN(bidirectional=False)
    cqt = np.random.default_rng(0).random((144, n_frames), dtype=np.float32)
    strip = np.pad(cqt, ((0, 0), (7, 7)))
    whole = model_v6.frame_posteriors(model, torch.from_numpy(strip)[None, None]).numpy()
    s = StreamingCRNN(model)
    live = [p for p in (s.step(cqt[:, t]) for t in range(n_frames)) if p is not None] + s.flush()
    return whole, np.stack(live, axis=1)


@pytest.mark.parametrize("n_frames", [40, 4])
def test_streaming_gives_the_whole_song_posteriors(n_frames):
    whole, live = posteriors_both_ways(n_frames)
    assert live.shape == whole.shape == (24, n_frames)
    np.testing.assert_allclose(live, whole, atol=1e-5)


def test_a_bidirectional_model_cannot_stream():
    with pytest.raises(ValueError):
        StreamingCRNN(model_v6.ChordCRNN(bidirectional=True))
