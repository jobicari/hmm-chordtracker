"""Component 39 (V13): V8's causal CRNN with a played-bass head read from the same GRU state.

model_v8.ChordCRNN ends in dropout and Linear(64 -> 60), the chord head. ChordCRNNBass keeps it and adds
Linear(64 -> 13) after the same dropout: the pitch class of the lowest bass note sounding at the frame
(0 = C .. 11 = B) or 12, no bass note. forward() still returns the chord logits only, so every decoder
and the distillation term are unchanged; heads() returns both. The extra layer is created after V8's
layers, so a V8 or V11 state dict loads into it with only the bass head left at its initial weights.

    ChordCRNNBass(n_classes=60, dropout=0.5)
    load(run, n_classes, cfg)     a saved V13 run: with the bass head when run/run.json says so
"""
import json
from pathlib import Path

import torch
from torch import nn

from chordtracker import model_v6, model_v8
from chordtracker.data_v5 import PATCH_FRAMES
from chordtracker.model_v5 import device

N_BASS = 13                 # 12 pitch classes + no bass note
NO_BASS = 12


class ChordCRNNBass(model_v8.ChordCRNN):
    """(B, 1, 144, L + 14) strips -> (B, L, n_classes) chord logits; heads() adds (B, L, 13) bass logits."""

    def __init__(self, n_classes: int = 60, dropout: float = 0.5):
        super().__init__(n_classes, bidirectional=False, dropout=dropout)
        self.bass = nn.Linear(model_v6.HIDDEN, N_BASS)

    def heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(chord, bass) logits, (B, L, n_classes) and (B, L, 13)."""
        b, length = x.shape[0], x.shape[3] - PATCH_FRAMES + 1
        e = self.frame_embeddings(model_v6.windows(x)).reshape(b, length, model_v6.EMBED)
        h = self.classifier[0](self.gru(e)[0])                       # dropout, shared by both heads
        return self.classifier[1](h), self.bass(h)


def from_chord_model(state: dict, n_classes: int, dropout: float) -> ChordCRNNBass:
    """A ChordCRNNBass holding a ChordCRNN's weights (e.g. V11's stage A); the bass head stays as initialised."""
    model = ChordCRNNBass(n_classes, dropout)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("bass.") for k in missing):
        raise ValueError(f"not a chord model's state: missing {missing}, unexpected {unexpected}")
    return model


def load(run: Path, n_classes: int, cfg: dict) -> nn.Module:
    """The model saved in run/model.pt, on device(), eval mode: ChordCRNNBass when run/run.json has bass_weight > 0."""
    info = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    if info.get("train", {}).get("bass_weight", 0):
        model = ChordCRNNBass(n_classes, dropout=cfg["train"]["dropout"])
    else:
        model = model_v8.ChordCRNN(n_classes, bidirectional=False, dropout=cfg["train"]["dropout"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model.to(device()).eval()
