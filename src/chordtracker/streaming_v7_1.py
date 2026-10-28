"""Component 22 (V7.1): the causal CRNN of V7.1 run one log-CQT column at a time.

V7's StreamingCRNN waits 7 columns before it can label a frame (V6's window is centred). V7.1's window
ends at its frame, so frame t is labelled as soon as column t arrives:

    step(column)  (144,) column of frame t -> (24,) P(chord | frame t)

A test checks that this equals train_v7_1.song_posteriors on the whole song.
"""
import numpy as np
import torch

from chordtracker.streaming_crnn import StreamingCRNN


class CausalStreamingCRNN(StreamingCRNN):
    """StreamingCRNN for a window that ends at its frame: no wait, no flush."""

    @torch.no_grad()
    def step(self, column: np.ndarray) -> np.ndarray:
        self.window = torch.cat([self.window[..., 1:], torch.from_numpy(column).float()[None, None, :, None]], 3)
        self.seen += 1
        out, self.hidden = self.model.gru(self.model.frame_embeddings(self.window)[None], self.hidden)
        return torch.softmax(self.model.classifier(out)[0, 0], dim=0).numpy()

    def flush(self) -> list[np.ndarray]:
        return []
