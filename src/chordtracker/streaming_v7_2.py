"""Component 25 (V7.2): the causal CRNN with an input look-ahead of a frames, one column at a time.

With a window that ends a frames after its frame, frame t is labelled when column t + a arrives:

    step(column t)  -> (24,) P(chord | frame t - a), or None while t < a
    flush()         the last a frames' posteriors at the end of the song (zero columns after it)

a = 7 is V7's StreamingCRNN (centred window), a = 0 is V7.1's CausalStreamingCRNN.
"""
import numpy as np
import torch

from chordtracker.data_v5 import PATCH_BINS
from chordtracker.streaming_crnn import StreamingCRNN


class LookaheadStreamingCRNN(StreamingCRNN):
    """StreamingCRNN for a window that reaches `lookahead` frames past its frame."""

    def __init__(self, model, lookahead: int):
        super().__init__(model)
        self.lookahead = lookahead

    @torch.no_grad()
    def step(self, column: np.ndarray) -> np.ndarray | None:
        self.window = torch.cat([self.window[..., 1:], torch.from_numpy(column).float()[None, None, :, None]], 3)
        self.seen += 1
        if self.seen <= self.lookahead:
            return None
        out, self.hidden = self.model.gru(self.model.frame_embeddings(self.window)[None], self.hidden)
        return torch.softmax(self.model.classifier(out)[0, 0], dim=0).numpy()

    def flush(self) -> list[np.ndarray]:
        out = [self.step(np.zeros(PATCH_BINS, dtype=np.float32)) for _ in range(self.lookahead)]
        return [p for p in out if p is not None]
