"""Tests for chordtracker.features_v2 and chordtracker.bass (components 8 and 9, V2).

Unit tests use small hand-made arrays and need no audio. The tests marked `slow` run the real
functions on one GuitarSet train take (no val or test take is touched): run them with `pytest -m slow`.
"""
import copy
from pathlib import Path

import librosa
import numpy as np
import pytest
import yaml

from chordtracker import bass, features_v2, templates
from chordtracker.emission import TemplateEmission

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "v2.yaml").read_text())
MANIFEST = ROOT / "data" / "processed" / "guitarset" / "manifest.csv"
FREQS = features_v2.bin_frequencies()
FLOOR = {"enabled": True, "percentile": 50, "alpha": 1.0, "fmax": None}


def bin_of(note: str) -> int:
    """CQT bin centred on a note (tuning 0): 3 bins per semitone from C1."""
    return 3 * (librosa.note_to_midi(note) - librosa.note_to_midi("C1"))


# ------------------------------------------------------------------------ floor

def test_median_floor_removes_a_constant_bin_and_keeps_a_short_note():
    """A resonance present in every frame is its own median; a note sounding in 3 of 10 frames survives."""
    C = np.zeros((252, 10))
    C[bin_of("A2")] = 2.0                      # constant: a body resonance
    C[bin_of("E3"), 2:5] = 1.0                 # a note held for 3 frames
    out = features_v2.remove_floor(C, FREQS, FLOOR)
    assert np.allclose(out[bin_of("A2")], 0.0)
    assert np.allclose(out[bin_of("E3")], C[bin_of("E3")])


def test_median_floor_removes_a_note_held_for_most_of_the_take():
    """The limit Jo asked about: a note in more than half of the frames is in its bin's median."""
    C = np.zeros((252, 10))
    C[bin_of("E3"), :7] = 1.0
    assert np.allclose(features_v2.remove_floor(C, FREQS, FLOOR)[bin_of("E3")], 0.0)
    gentle = dict(FLOOR, percentile=20)        # the 20th percentile keeps it: it is silent in 30% of frames
    assert np.allclose(features_v2.remove_floor(C, FREQS, gentle)[bin_of("E3")], C[bin_of("E3")])


def test_floor_only_changes_bins_below_fmax_and_never_goes_negative():
    C = np.ones((252, 10)) + np.arange(10)     # every bin rises over time
    out = features_v2.remove_floor(C, FREQS, dict(FLOOR, fmax=200.0))
    low, high = FREQS < 200.0, FREQS >= 200.0
    assert np.allclose(out[high], C[high])
    assert np.all(out[low] < C[low]) and np.all(out >= 0.0)


def test_floor_disabled_returns_the_input():
    C = np.random.default_rng(0).random((252, 8))
    assert features_v2.remove_floor(C, FREQS, dict(FLOOR, enabled=False)) is C


# ------------------------------------------------------------------------ compression

def test_log_compression_keeps_order_and_ignores_the_recording_level():
    """log1p is monotonic, and dividing by the take level makes a 10x louder take give the same result."""
    cfg = {"enabled": True, "level_percentile": 99, "gamma": 10}
    C = np.random.default_rng(1).random((252, 20))
    out = features_v2.log_compress(C, cfg)
    assert np.array_equal(np.argsort(out, axis=None), np.argsort(C, axis=None))
    assert np.allclose(features_v2.log_compress(10 * C, cfg), out)
    assert features_v2.log_compress(C, dict(cfg, enabled=False)) is C


# ------------------------------------------------------------------------ folding

def test_fold_chroma_equals_chroma_cqt_on_the_same_cqt():
    """With the V1 settings, fold_chroma(|CQT|) is chroma_cqt itself: the V2 path starts from V1."""
    y = librosa.tone(220.0, sr=22050, duration=1.0) + librosa.tone(330.0, sr=22050, duration=1.0)
    cfg = copy.deepcopy(CFG)
    cfg["features"]["threshold"] = 0.25
    C = features_v2.cqt_magnitude(y, 22050, 0.0, cfg)
    expected = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=512, threshold=0.25, tuning=0.0, norm=np.inf)
    assert np.allclose(features_v2.fold_chroma(C, cfg), expected)


def test_bass_chroma_ignores_energy_above_b3_and_sees_a_bass_guitar_low_e():
    C = np.zeros((252, 3))
    C[bin_of("E1"), 0] = 1.0                   # bass guitar low E (41 Hz): in range
    C[bin_of("G4"), 1] = 1.0                   # above B3: out of range
    C[bin_of("B3"), 2] = 1.0                   # top of the range: in
    out = features_v2.bass_chroma(C, FREQS, CFG["bass"])
    assert out[4, 0] == pytest.approx(1.0) and out[:, 0].sum() == pytest.approx(1.0)   # E
    assert np.allclose(out[:, 1], 0.0)
    assert out[11, 2] == pytest.approx(1.0)                                            # B


# ------------------------------------------------------------------------ bass term

def test_root_matrix_marks_each_root():
    roots = bass.root_matrix(0.25)
    labels = templates.majmin_labels()
    assert roots[labels.index("D:maj"), 2] == 1.0 and roots[labels.index("D:min"), 2] == 1.0
    assert roots[:24].sum(axis=1).tolist() == [1.0] * 24 and np.allclose(roots[24], 0.25)


def test_bass_weight_zero_is_the_template_emission():
    model = TemplateEmission.from_config(CFG)
    rng = np.random.default_rng(2)
    treble, bass_chroma = rng.random((12, 6)), rng.random((12, 6))
    assert np.allclose(bass.posteriors(model, treble, bass_chroma, 0.0), model.posteriors(treble))


def test_bass_note_breaks_a_tie_between_triads_sharing_two_notes():
    """F A in the treble fits D:min (D F A) and F:maj (F A C) equally; a D or an F in the bass decides."""
    model = TemplateEmission.from_config(CFG)
    labels = model.labels
    treble = np.zeros((12, 1)); treble[[5, 9]] = 1.0              # F, A
    for bass_pc, winner in [(2, "D:min"), (5, "F:maj")]:
        bass_chroma = np.zeros((12, 1)); bass_chroma[bass_pc] = 1.0
        post = bass.posteriors(model, treble, bass_chroma, 0.2)[:, 0]
        assert labels[int(post.argmax())] == winner
        assert np.isclose(post.sum(), 1.0)


# ------------------------------------------------------------------------ real audio

@pytest.mark.slow
@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest not found in data/processed/guitarset")
def test_real_train_take_v2_off_equals_v1_and_v2_runs():
    """On train take 00_BN1-129-Eb_comp: with every V2 step off the treble is the V1 chroma; V2 evidence is valid."""
    from chordtracker.data.data import load_manifest
    from chordtracker.features import ChromaCQTExtractor

    record = next(r for r in load_manifest(MANIFEST) if r.song_id == "00_BN1-129-Eb_comp")
    v1_cfg = yaml.safe_load((ROOT / "configs" / "v1.yaml").read_text())
    off = copy.deepcopy(CFG)
    off["features"]["threshold"] = v1_cfg["features"]["threshold"]
    off["floor"]["enabled"] = off["compression"]["enabled"] = False
    treble, bass_chroma = features_v2.extract(record.audio, off)
    assert np.allclose(treble, ChromaCQTExtractor.from_config(v1_cfg).extract(record.audio), atol=1e-6)
    assert bass_chroma.shape == treble.shape and np.all((0.0 <= bass_chroma) & (bass_chroma <= 1.0))

    take = bass.take_evidence(record, TemplateEmission.from_config(CFG), CFG)
    assert take.posteriors.shape == (25, len(take.edges) - 1)
    assert np.allclose(take.posteriors.sum(axis=0), 1.0)
    assert take.edges[0] == 0.0 and take.edges[-1] == take.duration
