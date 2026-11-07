"""Component 36 (V10): V8's causal CRNN with three extra output heads read from the same GRU state.

model_v8.ChordCRNN ends in dropout and Linear(64 -> 60), the chord head. ChordCRNNHeads keeps it and adds
one Linear(64 -> 12 + 5 + 12) after the same dropout:

    root     12 pitch classes, the chord's root
    quality  5 qualities of vocabulary A (maj, min, 7, maj7, min7)
    bass     12 pitch classes, the lowest note of the Harte label (C:maj/3 -> E)

forward() still returns the chord logits only, so model_v6.frame_posteriors and every decoder are
unchanged; heads() returns all four. The extra layer is created after V8's layers, so with the same seed
the shared layers start from V8's initial weights.

    ChordCRNNHeads(n_classes=60, n_qualities=5, dropout=0.5)
    load(run, n_classes, cfg)     a saved V8 or V10 run (run.json says whether it has the heads)
"""
import json
from pathlib import Path

import torch
from torch import nn

from chordtracker import model_v6, model_v8
from chordtracker.data_v5 import PATCH_FRAMES
from chordtracker.model_v5 import device

N_ROOTS = 12


class ChordCRNNHeads(model_v8.ChordCRNN):
    """(B, 1, 144, L + 14) strips -> (B, L, n_classes) chord logits; heads() adds root, quality and bass."""

    def __init__(self, n_classes: int = 60, n_qualities: int = 5, dropout: float = 0.5):
        super().__init__(n_classes, bidirectional=False, dropout=dropout)
        self.n_qualities = n_qualities
        self.aux = nn.Linear(model_v6.HIDDEN, 2 * N_ROOTS + n_qualities)

    def heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(chord, root, quality, bass) logits, each (B, L, n)."""
        b, length = x.shape[0], x.shape[3] - PATCH_FRAMES + 1
        e = self.frame_embeddings(model_v6.windows(x)).reshape(b, length, model_v6.EMBED)
        h = self.classifier[0](self.gru(e)[0])                       # dropout, shared by every head
        aux = self.aux(h)
        q = self.n_qualities
        return self.classifier[1](h), aux[..., :N_ROOTS], aux[..., N_ROOTS:N_ROOTS + q], aux[..., N_ROOTS + q:]


def load(run: Path, n_classes: int, cfg: dict) -> nn.Module:
    """The model saved in run/model.pt, on device(), eval mode: with heads when run/run.json says so."""
    info = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    if info.get("heads"):
        model = ChordCRNNHeads(n_classes, info["n_qualities"], dropout=cfg["train"]["dropout"])
    else:
        model = model_v8.ChordCRNN(n_classes, bidirectional=False, dropout=cfg["train"]["dropout"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model.to(device()).eval()
