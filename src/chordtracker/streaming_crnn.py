"""Component 18 (V7): the causal V6 CRNN run one CQT frame at a time, as a live system would run it.

model_v6.frame_posteriors reads a whole song at once. A live system gets one log-CQT column every
93 ms instead. StreamingCRNN keeps what it needs between columns:

    the last 15 columns     the window of the frame 7 columns back (data_v5.patch is centred, so a
                            frame's window is complete only 7 frames after the frame itself)
    the GRU state           64 numbers that carry everything older than the window

    step(column)  (144,) column of frame t -> (24,) P(chord | frame t - 7), or None for t < 7
    flush()       the last 7 frames' posteriors at the end of the song (zero columns after it, as
                  data_v6.excerpt pads)

Only a forward GRU can run this way; a test checks that the posteriors equal frame_posteriors on
the whole song, so the frame-level results of run_v7 are exactly what this loop would produce live.
"""
import numpy as np
import torch

from chordtracker.data_v5 import PATCH_BINS, PATCH_FRAMES
from chordtracker.model_v6 import ChordCRNN

MARGIN = PATCH_FRAMES // 2       # 7 frames: how far the window of a frame reaches past it


class StreamingCRNN:
    """One causal ChordCRNN, fed one (144,) log-CQT column per step; eval mode, CPU."""

    def __init__(self, model: ChordCRNN):
        if model.gru.bidirectional:
            raise ValueError("a bidirectional GRU needs the whole song; stream a causal model")
        self.model = model.eval().cpu()
        self.window = torch.zeros(1, 1, PATCH_BINS, PATCH_FRAMES)     # zeros before the song, as excerpt
        self.hidden = None
        self.seen = 0

    @torch.no_grad()
    def step(self, column: np.ndarray) -> np.ndarray | None:
        """Add the column of frame t; return P(chord | frame t - 7), or None while t < 7."""
        self.window = torch.cat([self.window[..., 1:], torch.from_numpy(column).float()[None, None, :, None]], 3)
        self.seen += 1
        if self.seen <= MARGIN:
            return None
        embedding = self.model.frame_embeddings(self.window)[None]   # (1, 1, 64)
        out, self.hidden = self.model.gru(embedding, self.hidden)
        return torch.softmax(self.model.classifier(out)[0, 0], dim=0).numpy()

    def flush(self) -> list[np.ndarray]:
        """The posteriors of the last 7 frames (or all frames of a song shorter than that)."""
        out = []
        for _ in range(MARGIN):
            p = self.step(np.zeros(PATCH_BINS, dtype=np.float32))
            if p is not None:
                out.append(p)
        return out
