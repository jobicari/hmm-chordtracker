"""Component 31 (V8): V7.3's causal CRNN with as many outputs as the vocabulary has classes.

model_v6.ChordCRNN ends in Linear(64 -> 24). V8 keeps every other layer and gives the last one
n_classes outputs (data_v8: 60 for A, 84 for B, 96 for B9). With n_classes = 24 the model is
model_v6's, weight for weight, so V7.3's weights still load.

    ChordCRNN(n_classes, bidirectional=False, dropout=0.5)
    n_weights(model)

Weights: 99,152 at 24 classes; each class adds 65 (64 weights and a bias), so 101,492 at 60.
model_v6.frame_posteriors works unchanged: it returns (n_classes, n) posteriors.
"""
from torch import nn

from chordtracker import model_v6


class ChordCRNN(model_v6.ChordCRNN):
    """(B, 1, 144, L + 14) strips -> (B, L, n_classes) chord logits per frame, in data_v8 class order."""

    def __init__(self, n_classes: int = 60, bidirectional: bool = False, dropout: float = 0.5):
        super().__init__(bidirectional=bidirectional, dropout=dropout)
        self.n_classes = n_classes
        if n_classes != model_v6.N_CLASSES:                 # at 24, keep model_v6's layer and its initialisation
            last = self.classifier[1]
            self.classifier[1] = nn.Linear(last.in_features, n_classes)


def n_weights(model: nn.Module) -> int:
    """Number of trained numbers in the model."""
    return sum(p.numel() for p in model.parameters())
