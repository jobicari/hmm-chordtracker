"""Component 15 (V6): the acoustic model with temporal context, a CRNN from a log-CQT strip to 24 chord scores per frame.

V5's CNN labels each frame from the 1.4 s around it and nothing else. The CRNN keeps that front end
and adds a recurrent layer (GRU) that reads the frames in order, so the score of a frame can depend
on the chords before it (causal) or before and after it (bidirectional). The shapes, for a batch of
B excerpts of L frames (data_v6.excerpt: L + 14 columns, 7 frames of context on each side):

    input                          B x 1 x 144 x (L + 14)
       unfold                      B*L x 1 x 144 x 15      the 15-frame window of every frame, as V5's patch
    A, B, C  V5's conv blocks      B*L x 64 x 72           same layers as model_v5, mean over time
    E  conv 1 x 1, 64 -> 8         B*L x 8 x 72            8 numbers per semitone row, 576 in all
       dense 576 -> 64, ReLU       B x L x 64              one embedding per frame
    G  GRU, hidden 64              B x L x 128 (64 causal) forward pass, plus a backward pass if bidirectional
    D  dropout 0.5, dense          B x L x 24              one score (logit) per chord per frame

Why the 1 x 1 convolution: V5's dense layer reads all 64 x 72 = 4,608 numbers of a frame, 110k of
its 146k weights. Feeding that to a recurrent layer would put most of the model in one matrix again.
Mixing the 64 channels down to 8 per semitone row keeps WHERE a pattern is (the 72 rows: root and
octave) and shrinks the embedding's input to 576.

The front end sees exactly V5's numbers: the window of frame t is columns t .. t + 14 of the strip,
which is data_v5.patch(song, start + t, k). So any change in the result is the GRU's.
A whole song is one strip at inference (frame_posteriors): the conv runs in chunks of frames, then
the GRU makes one pass over the song.
"""
import torch
from torch import nn

from chordtracker.data_v5 import PATCH_BINS, PATCH_FRAMES
from chordtracker.model_v5 import conv_block

N_CLASSES = 24
EMBED = 64                      # numbers per frame that the GRU reads
HIDDEN = 64                     # GRU state per direction
ROW_CHANNELS = 8                # channels per semitone row after the 1 x 1 convolution


class ChordCRNN(nn.Module):
    """(B, 1, 144, L + 14) strips -> (B, L, 24) chord logits per frame, C:maj..B:maj, C:min..B:min."""

    def __init__(self, bidirectional: bool = True, dropout: float = 0.5):
        super().__init__()
        self.front = nn.Sequential(                                             # V5's A, B, C
            conv_block(1, 16), conv_block(16, 16), nn.MaxPool2d((2, 1)),
            conv_block(16, 32), conv_block(32, 32), nn.MaxPool2d((1, 3)),
            conv_block(32, 64),
        )
        self.mix = nn.Conv2d(64, ROW_CHANNELS, 1)                               # E
        self.embed = nn.Sequential(nn.Linear(ROW_CHANNELS * PATCH_BINS // 2, EMBED), nn.ReLU())
        self.gru = nn.GRU(EMBED, HIDDEN, batch_first=True, bidirectional=bidirectional)    # G
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(HIDDEN * (2 if bidirectional else 1), N_CLASSES))   # D

    def frame_embeddings(self, patches: torch.Tensor) -> torch.Tensor:
        """(n, 1, 144, 15) windows -> (n, 64): the per-frame front end, no recurrence."""
        h = self.front(patches).mean(dim=3, keepdim=True)                       # (n, 64, 72, 1)
        return self.embed(self.mix(h).flatten(1))

    def sequence(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(B, L, 64) frame embeddings in order -> (B, L, 24) logits."""
        return self.classifier(self.gru(embeddings)[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length = x.shape[0], x.shape[3] - PATCH_FRAMES + 1
        return self.sequence(self.frame_embeddings(windows(x)).reshape(b, length, EMBED))


def windows(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, 144, L + 14) strip -> (B * L, 1, 144, 15): the window of every frame, frames of excerpt 0 first."""
    w = x.unfold(3, PATCH_FRAMES, 1)                                            # (B, 1, 144, L, 15)
    return w.permute(0, 3, 1, 2, 4).reshape(-1, 1, x.shape[2], PATCH_FRAMES)


@torch.no_grad()
def frame_posteriors(model: ChordCRNN, strip: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """(1, 1, 144, n + 14) strip of a whole song -> (24, n) posteriors P(chord | frame), eval mode.

    The conv front end runs on `chunk` frames at a time; the GRU then reads the whole song at once.
    The (24, n) layout is acoustic_hmm.frames_to_beats's, as model_v5.frame_posteriors.
    """
    model.eval()
    dev = next(model.parameters()).device
    w = windows(strip)
    e = torch.cat([model.frame_embeddings(w[i:i + chunk].to(dev)) for i in range(0, len(w), chunk)])
    return torch.softmax(model.sequence(e[None]), dim=2)[0].cpu().T
