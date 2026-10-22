"""Component 3b: emission models, the HMM's evidence term, from frame features to chord posteriors.

The textbook HMM emission is the likelihood P(frame | state). The models here output the other
direction, the posterior P(state | frame): a (n_states, n_frames) matrix whose columns sum to 1,
whose argmax per column is the framewise decision. librosa.sequence.viterbi_discriminative turns
it back into an emission by dividing by a state prior p_state (Bayes: P(frame | state) is
proportional to P(state | frame) / P(state)) before running Viterbi. V0 to V3 use a uniform
p_state, so the division changes nothing; V4 divides by the train class frequencies.

V0 has one model, TemplateEmission. Learned models (V4, V5) will be further subclasses.
"""
from abc import ABC, abstractmethod

import numpy as np

from chordtracker import templates as tpl


class EmissionModel(ABC):
    """Turns a (n_features, n_frames) matrix into (n_states, n_frames) posteriors. Subclasses implement posteriors().

    labels[i] is the chord label of state i, so a decoded state path maps to labels with
    [model.labels[i] for i in path].
    """
    labels: list[str]

    @abstractmethod
    def posteriors(self, features: np.ndarray) -> np.ndarray:
        """ P(state | frame) for every frame: shape (n_states, n_frames), each column sums to 1 """




class TemplateEmission(EmissionModel):
    """V0 model: score every chord template against the chroma, then softmax over the states.

        scores     = templates @ chroma               (25, 12) @ (12, T) -> (25, T)
        posteriors = softmax(beta * scores) per column

    Example on one frame with a pure C major chroma (1 on C, E, G, 0 elsewhere):
    C:maj scores 3, A:min and E:min score 2 (two shared notes), N scores 0.25 * 3 = 0.75.
    beta sets how sharp the softmax is: beta = 1 is the notebook and the librosa example;
    a larger beta pushes the posteriors towards 0/1, a smaller one towards uniform.
    """

    def __init__(self, templates: np.ndarray, labels: list[str], beta=1.0):
        if templates.shape[0] != len(labels):
            raise ValueError(f"{templates.shape[0]} templates for {len(labels)} labels")
        self.templates = templates      # (n_states, 12)
        self.labels = labels
        self.beta = beta

    @classmethod
    def from_config(cls, cfg: dict):
        """Build the model from a loaded YAML config (cfg["templates"] and cfg["emission"])."""
        templates_cfg, emission_cfg = cfg["templates"], cfg["emission"]
        if templates_cfg["vocab"] != "majmin":
            raise ValueError(f"unsupported vocabulary {templates_cfg['vocab']!r}, expected 'majmin'")
        return cls(tpl.majmin_templates(templates_cfg["no_chord_value"]), tpl.majmin_labels(),
                   beta=emission_cfg["beta"])

    def scores(self, chroma: np.ndarray) -> np.ndarray:
        """How well each template fits each frame: (n_states, 12) @ (12, n_frames) -> (n_states, n_frames)."""
        return self.templates @ chroma

    def posteriors(self, chroma):
        scaled_scores = self.beta * self.scores(chroma)
        # subtracting each column's max changes nothing after normalizing (exp(a - m) / sum = exp(a) / sum)
        # but keeps exp() from overflowing when beta is large
        scaled_scores -= scaled_scores.max(axis=0, keepdims=True)
        unnormalized = np.exp(scaled_scores)
        return unnormalized / unnormalized.sum(axis=0, keepdims=True)
