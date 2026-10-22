"""Component 4: decoders, from chord posteriors to one chord state per frame.

A decoder takes the (n_states, n_frames) posteriors P(state | frame) output by an emission model
(see emission.py: the posterior, not the textbook likelihood) and returns a state path
of shape (n_frames,): path[t] is the index of the chord chosen at frame t, so the labels are
[model.labels[i] for i in path]. The two V0 decoders are the two systems the project compares:

    FramewiseDecoder    every frame on its own: argmax of its column
    ViterbiDecoder      the whole sequence at once: the most likely path under a transition model
                        that stays on the current chord with probability p_self

Both read the same posteriors, so any difference between them comes from the temporal model only.
"""
from abc import ABC, abstractmethod

import librosa
import numpy as np


# ====================================================================================
# ================================     DECODER      ==================================
# ====================================================================================

class Decoder(ABC):
    """Turns (n_states, n_frames) posteriors into a (n_frames,) state path. Subclasses implement decode().

    Example: 25 states and 3 frames in, an integer array such as [3, 3, 15] out
    (D#:maj, D#:maj, D#:min with the majmin labels).
    """

    @abstractmethod
    def decode(self, posteriors: np.ndarray) -> np.ndarray:
        """ The chosen state per frame: shape (n_frames,), integer values in 0..n_states - 1 """


# ====================================================================================
# ================================     FRAMEWISE    ==================================
# ====================================================================================

class FramewiseDecoder(Decoder):
    """Baseline: pick the most probable state in every frame, ignoring its neighbours.

    Example with 3 states and 2 frames:
        posteriors = [[0.6, 0.2],
                      [0.3, 0.5],
                      [0.1, 0.3]]   ->   path = [0, 1]
    A tie goes to the lowest state index (np.argmax returns the first maximum).
    """

    @classmethod
    def from_config(cls, cfg: dict):
        """Framewise has no settings; from_config exists so every decoder is built the same way."""
        return cls()

    def decode(self, posteriors):
        return np.argmax(posteriors, axis=0)


# ====================================================================================
# ================================      VITERBI     ==================================
# ====================================================================================

class ViterbiDecoder(Decoder):
    """HMM decoder: the most likely state path when every chord stays with probability p_self.

    The transition matrix is librosa.sequence.transition_loop(n_states, p_self):
        A[i, i] = p_self                         stay on chord i
        A[i, j] = (1 - p_self) / (n_states - 1)  move to any other chord j, all equally likely
    With 25 states and p_self = 0.9: A[i, i] = 0.9 and A[i, j] = 0.1 / 24 = 0.00417.
    A chord then lasts 1 / (1 - p_self) frames on average: 10 frames = 0.23 s at hop 512 and 22050 Hz.

    librosa.sequence.viterbi_discriminative divides the posteriors by the state marginal (uniform
    here, so it changes nothing) and runs Viterbi in log space. Switching chord costs about
    log(p_self / ((1 - p_self) / 24)) in log-probability, about 7.8 at p_self = 0.99, so a change only
    happens when the new chord wins by that much summed over the frames it covers.

    p_self = 1 / n_states makes every transition equal (0.04 for 25 states), and Viterbi then gives
    exactly the framewise path.
    """

    def __init__(self, p_self: float):
        if not 0.0 < p_self < 1.0:
            raise ValueError(f"p_self must be strictly between 0 and 1, got {p_self}")
        self.p_self = p_self

    @classmethod
    def from_config(cls, cfg: dict):
        """Build the decoder from a loaded YAML config: p_self = cfg["decoder"]["p_self"] (0.9 in v0.yaml)."""
        return cls(cfg["decoder"]["p_self"])

    def transition(self, n_states: int) -> np.ndarray:
        """The (n_states, n_states) transition matrix; each row sums to 1. Example above for 25 states."""
        return librosa.sequence.transition_loop(n_states, self.p_self)

    def decode(self, posteriors):
        return librosa.sequence.viterbi_discriminative(posteriors, self.transition(posteriors.shape[0]))
