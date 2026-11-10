"""Component 38 (V14): V8's causal CRNN with an N class, and optionally the input level as one more GRU input.

    ChordCRNN(n_classes=61, level_input=False, dropout=0.5)
    frame_posteriors(model, strip, level=None)       (n_classes, n) posteriors of a whole song
    StreamingCRNN(model, lookahead)                  the same one column (and one level) at a time
    load(run, cfg)                                   a saved V14 run (run.json: n_classes, level_input)

Without the level this is model_v8.ChordCRNN(n_classes) layer for layer (61 outputs: vocabulary A and N),
created in the same order, so a seed starts from the same weights as V8's but for the extra class.

With level_input the CNN is unchanged and the GRU reads 65 numbers per frame: the 64 of the embedding
and the frame's scaled dBFS level (features_v14). The GRU step of frame t reads the window of columns up
to t + a (input look-ahead a), so it is given the level of frame t + a, the newest frame it has heard:

    level strip (B, L)    level[t] = features_v14 level of frame start + t + a, -1 (silence) outside the song
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from chordtracker import model_v6, model_v8
from chordtracker.data_v5 import PATCH_BINS, PATCH_FRAMES
from chordtracker.model_v5 import device

SILENCE = -1.0                   # the scaled level of digital silence (features_v14.FLOOR_DB)


class ChordCRNN(model_v8.ChordCRNN):
    """(B, 1, 144, L + 14) strips [and (B, L) levels] -> (B, L, n_classes) logits per frame, N last."""

    def __init__(self, n_classes: int = 61, level_input: bool = False, dropout: float = 0.5):
        super().__init__(n_classes, bidirectional=False, dropout=dropout)
        self.level_input = level_input
        if level_input:
            self.gru = nn.GRU(model_v6.EMBED + 1, model_v6.HIDDEN, batch_first=True)

    def gru_input(self, embeddings: torch.Tensor, level: torch.Tensor | None) -> torch.Tensor:
        """(B, L, 64) embeddings -> what the GRU reads: the same, or with the (B, L) level appended."""
        if not self.level_input:
            return embeddings
        if level is None:
            raise ValueError("this model reads the input level: pass level")
        return torch.cat([embeddings, level.to(embeddings)[..., None]], dim=2)

    def forward(self, x: torch.Tensor, level: torch.Tensor | None = None) -> torch.Tensor:
        b, length = x.shape[0], x.shape[3] - PATCH_FRAMES + 1
        e = self.frame_embeddings(model_v6.windows(x)).reshape(b, length, model_v6.EMBED)
        return self.classifier(self.gru(self.gru_input(e, level))[0])


@torch.no_grad()
def frame_posteriors(model: ChordCRNN, strip: torch.Tensor, level: torch.Tensor | None = None,
                     chunk: int = 1024) -> torch.Tensor:
    """(1, 1, 144, n + 14) strip [and (1, n) level] of a whole song -> (n_classes, n) posteriors, eval mode."""
    model.eval()
    dev = next(model.parameters()).device
    w = model_v6.windows(strip)
    e = torch.cat([model.frame_embeddings(w[i:i + chunk].to(dev)) for i in range(0, len(w), chunk)])[None]
    h = model.gru(model.gru_input(e, None if level is None else level.to(dev)))[0]
    return torch.softmax(model.classifier(h), dim=2)[0].cpu().T


class StreamingCRNN:
    """A causal ChordCRNN fed one (144,) log-CQT column (and that frame's level) per step; eval mode, CPU.

    step(column t, level t) -> (n_classes,) P(class | frame t - a), or None while t < a; flush() gives the
    last a frames (zero columns and silence after the song, as the offline strip pads).
    """

    def __init__(self, model: ChordCRNN, lookahead: int):
        self.model = model.eval().cpu()
        self.lookahead = lookahead
        self.window = torch.zeros(1, 1, PATCH_BINS, PATCH_FRAMES)     # zeros before the song, as the excerpt
        self.hidden = None
        self.seen = 0

    @torch.no_grad()
    def step(self, column: np.ndarray, level: float | None = None) -> np.ndarray | None:
        self.window = torch.cat([self.window[..., 1:], torch.from_numpy(column).float()[None, None, :, None]], 3)
        self.seen += 1
        if self.seen <= self.lookahead:
            return None
        e = self.model.frame_embeddings(self.window)[None]                   # (1, 1, 64)
        lv = None if level is None else torch.tensor([[float(level)]])
        out, self.hidden = self.model.gru(self.model.gru_input(e, lv), self.hidden)
        return torch.softmax(self.model.classifier(out)[0, 0], dim=0).numpy()

    def flush(self) -> list[np.ndarray]:
        level = SILENCE if self.model.level_input else None
        out = [self.step(np.zeros(PATCH_BINS, dtype=np.float32), level) for _ in range(self.lookahead)]
        return [p for p in out if p is not None]


def load(run: Path, cfg: dict) -> ChordCRNN:
    """The model saved in run/model.pt, on device(), eval mode, shaped by run/run.json."""
    info = json.loads((run / "run.json").read_text())
    model = ChordCRNN(info["n_classes"], level_input=info["level_input"], dropout=cfg["train"]["dropout"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model.to(device()).eval()
