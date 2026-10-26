"""V5.1: V5's CNN with a 25th output, N (no chord). Blocks A to C are model_v5's, unchanged.

    input -> A, B, C (model_v5.ChordCNN)  B x 64 x 72      as in V5
    D  dropout 0.5, dense                 B x 25           C:maj..B:maj, C:min..B:min, N

The only new weights are one more row of the dense layer: 4,608 + 1 = 4,609, so 150,409 in all
(V5: 145,800). Why N can be a single extra output: silence and fade-outs look the same in every key,
so N needs no root and no copy per key, unlike the 24 chords.

The network is trained from scratch, not started from V5's weights, so the only difference from V5's
training run is the 25th class and the frames it brings into the loss.
model_v5.frame_posteriors and device() work as they are and return (25, n) posteriors here.
"""
from torch import nn

from chordtracker.data_v5 import PATCH_BINS
from chordtracker.data_v5_1 import N_CLASSES
from chordtracker.model_v5 import ChordCNN


class ChordCNNWithN(ChordCNN):
    """(B, 1, 144, 15) log-CQT patches -> (B, 25) logits: V5's 24 chords, then N."""

    def __init__(self, dropout: float = 0.5):
        super().__init__(dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(64 * PATCH_BINS // 2, N_CLASSES))
