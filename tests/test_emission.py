"""Tests for chordtracker.templates and chordtracker.emission (component 3).

Unit tests use hand-made chroma (a template fed back as chroma, or a few frames of known notes),
so the expected state is known exactly. The test marked `slow` reproduces the V0 notebook's
framewise numbers on a real GuitarSet take: run it with `pytest -m slow`.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from chordtracker import emission, templates as tpl

GUITARSET = Path(__file__).resolve().parents[1] / "data" / "guitarset"
CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "v0.yaml").read_text())
LABELS = tpl.majmin_labels()
TEMPLATES = tpl.majmin_templates()


# ------------------------------------------------------------------------ templates

def test_labels_order():
    """25 labels: majors at 0..11, minors at 12..23, N last; GuitarSet's D#:maj is state 3."""
    assert len(LABELS) == 25
    assert (LABELS[0], LABELS[3], LABELS[12], LABELS[21], LABELS[24]) == ("C:maj", "D#:maj", "C:min", "A:min", "N")


def test_templates_hold_the_right_notes():
    """Each row has 1 on its chord's pitch classes: E:maj = E G# B (4, 8, 11), A:min = A C E (9, 0, 4)."""
    assert TEMPLATES.shape == (25, 12)
    assert np.flatnonzero(TEMPLATES[LABELS.index("E:maj")]).tolist() == [4, 8, 11]
    assert np.flatnonzero(TEMPLATES[LABELS.index("A:min")]).tolist() == [0, 4, 9]
    assert (TEMPLATES[:24].sum(axis=1) == 3).all()                  # every triad has 3 notes
    assert (TEMPLATES[24] == CFG["templates"]["no_chord_value"]).all()


# ------------------------------------------------------------------------ emission

def test_emission_without_posteriors_cannot_be_created():
    """A subclass that forgets posteriors() raises TypeError at creation, thanks to @abstractmethod."""
    class Incomplete(emission.EmissionModel):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_template_emission_from_config():
    """from_config builds the 25-state majmin model with beta from cfg["emission"]; another vocab raises."""
    model = emission.TemplateEmission.from_config(CFG)
    assert model.labels == LABELS and model.beta == CFG["emission"]["beta"]
    np.testing.assert_array_equal(model.templates, TEMPLATES)
    with pytest.raises(ValueError):
        emission.TemplateEmission.from_config({**CFG, "templates": {**CFG["templates"], "vocab": "sevenths"}})


def test_template_emission_rejects_mismatched_labels():
    """25 templates with 24 labels raises ValueError when the model is created."""
    with pytest.raises(ValueError):
        emission.TemplateEmission(TEMPLATES, LABELS[:24])


def test_scores_on_a_c_major_frame():
    """Pure C major chroma (C, E, G = 1): C:maj scores 3, A:min and E:min 2, N 0.25 * 3 = 0.75."""
    model = emission.TemplateEmission.from_config(CFG)
    scores = model.scores(TEMPLATES[0][:, None])[:, 0]
    assert scores[LABELS.index("C:maj")] == 3
    assert scores[LABELS.index("A:min")] == scores[LABELS.index("E:min")] == 2
    assert scores[LABELS.index("N")] == 0.75


def test_posteriors_are_distributions_that_pick_each_chord():
    """Each column sums to 1, and feeding chord i's template as chroma makes state i the argmax (i = 0..23)."""
    model = emission.TemplateEmission.from_config(CFG)
    posteriors = model.posteriors(TEMPLATES[:24].T)                 # (25, 24): frame i is chord i
    np.testing.assert_allclose(posteriors.sum(axis=0), 1.0)
    assert posteriors.argmax(axis=0).tolist() == list(range(24))


def test_posteriors_match_the_notebook_formula_and_survive_large_beta():
    """beta = 1 equals the notebook's exp(scores) / sum; beta = 1000 gives finite values, no overflow."""
    chroma = np.random.default_rng(0).random((12, 50))
    model = emission.TemplateEmission(TEMPLATES, LABELS, beta=1.0)
    notebook = np.exp(TEMPLATES @ chroma)
    np.testing.assert_allclose(model.posteriors(chroma), notebook / notebook.sum(axis=0, keepdims=True))
    sharp = emission.TemplateEmission(TEMPLATES, LABELS, beta=1000.0).posteriors(chroma)
    assert np.isfinite(sharp).all()


def test_larger_beta_is_sharper():
    """The winning state's probability grows with beta: beta 0.1 < 1 < 10 on the same frame."""
    chroma = TEMPLATES[0][:, None]
    top = [emission.TemplateEmission(TEMPLATES, LABELS, beta=b).posteriors(chroma)[0, 0] for b in (0.1, 1.0, 10.0)]
    assert top[0] < top[1] < top[2]


# ------------------------------------------------------------------------ real data

@pytest.mark.slow
@pytest.mark.skipif(not GUITARSET.is_dir(), reason="GuitarSet not found in data/guitarset")
def test_real_take_framewise_matches_the_v0_notebook():
    """On take 00_BN1-129-Eb_comp, framewise argmax gives the notebook's 155.88 changes/min and 43.24%.

    Accuracy here is the notebook's frame accuracy (share of frames whose label equals the lead-sheet
    chord at the frame's centre time), not yet the mir_eval WCSR of the evaluation component.
    """
    from chordtracker.data.data import ChordAnnotation
    from chordtracker.features import ChromaCQTExtractor

    take = "00_BN1-129-Eb_comp"
    extractor = ChromaCQTExtractor.from_config(CFG)
    chroma = extractor.extract(GUITARSET / "audio_mono-mic_comp" / f"{take}_mic.wav")
    reference = ChordAnnotation.from_jams(GUITARSET / "annotation" / f"{take}.jams", index=0)

    model = emission.TemplateEmission.from_config(CFG)
    framewise_path = model.posteriors(chroma).argmax(axis=0)
    framewise_labels = [model.labels[i] for i in framewise_path]
    reference_labels = reference.at(extractor.frame_times(chroma.shape[1]))

    duration_minutes = reference.intervals[-1, 1] / 60                  # 22.3244 s, the audio length
    changes_per_minute = np.count_nonzero(np.diff(framewise_path)) / duration_minutes
    frame_accuracy = np.mean([est == ref for est, ref in zip(framewise_labels, reference_labels)])
    assert round(changes_per_minute, 2) == 155.88
    assert round(100 * frame_accuracy, 2) == 43.24
