"""Component 15 (V5): the acoustic model, a small CNN from a 144 x 15 log-CQT patch to 24 chord scores.

Every convolution is 3 x 3 (3 bins of pitch x 3 frames of time), followed by batch
norm and ReLU. The shapes, for a batch of B patches:

    input                         B x  1 x 144 x 15    144 bins = 6 octaves at 2 bins per semitone
    A  conv 16, conv 16           B x 16 x 144 x 15    local pitch patterns: a note, its neighbour bin
       max-pool 2 x 1 (pitch)     B x 16 x  72 x 15    one row per semitone, time untouched
    B  conv 32, conv 32           B x 32 x  72 x 15    intervals: thirds and fifths span 3-7 rows
       max-pool 1 x 3 (time)      B x 32 x  72 x  5    keep 1 frame in 3: the chord changes slowly
    C  conv 64                    B x 64 x  72 x  5
       mean over time             B x 64 x  72         what holds across the 1.4 s window
    D  dropout 0.5, dense         B x 24               4,608 numbers -> one score (logit) per chord

Pitch is never pooled below one row per semitone, and the dense layer sees all 72 semitone rows: the
network has to know WHERE a pattern is (which root, which octave), not only that it exists.
The model returns logits; softmax(logits) is P(chord | patch), the frame posterior acoustic_hmm pools.
About 146k weights, 110k of them in the dense layer.
"""
import torch
from torch import nn

from chordtracker.data_v5 import PATCH_BINS

N_CLASSES = 24


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """3 x 3 convolution (same size out) -> batch norm -> ReLU."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU())


class ChordCNN(nn.Module):
    """(B, 1, 144, 15) log-CQT patches -> (B, 24) chord logits, C:maj..B:maj, C:min..B:min."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16), conv_block(16, 16), nn.MaxPool2d((2, 1)),        # A
            conv_block(16, 32), conv_block(32, 32), nn.MaxPool2d((1, 3)),       # B
            conv_block(32, 64),                                                 # C
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(64 * PATCH_BINS // 2, N_CLASSES))  # D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).mean(dim=3)                    # mean over the 5 remaining frames
        return self.classifier(h.flatten(1))


def n_weights(model: nn.Module) -> int:
    """Number of trained numbers in the model (batch-norm running means are not counted)."""
    return sum(p.numel() for p in model.parameters())


def device() -> torch.device:
    """The Mac's GPU (MPS) when there is one, else the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def frame_posteriors(model: nn.Module, patches: torch.Tensor, batch: int = 1024) -> torch.Tensor:
    """(n, 1, 144, 15) patches -> (24, n) posteriors P(chord | frame), in eval mode (no dropout).

    The (24, n) layout is acoustic_hmm.frames_to_beats's: one column per frame, as V4's beat posteriors.
    """
    model.eval()
    dev = next(model.parameters()).device
    out = [torch.softmax(model(patches[i:i + batch].to(dev)), dim=1).cpu() for i in range(0, len(patches), batch)]
    return torch.cat(out).T
