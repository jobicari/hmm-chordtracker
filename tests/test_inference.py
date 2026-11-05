"""Behaviour of the packaged V6 inference: the frozen settings and weights, and the posteriors of a file."""
import hashlib

import numpy as np
import pytest
import soundfile as sf
import yaml

from chordtracker import inference, paths


def v6_config() -> dict:
    """configs/v6.yaml on top of its base, as scripts/run_v5.load_config reads it."""
    cfg = yaml.safe_load((paths.REPO_ROOT / "configs" / "v6.yaml").read_text())
    return {**yaml.safe_load((paths.REPO_ROOT / "configs" / cfg["base_config"]).read_text()), **cfg}


def test_settings_are_v6s():
    cfg = v6_config()
    for section in ("audio", "cqt", "compression"):
        assert inference.SETTINGS[section] == cfg[section]


def test_weights_are_the_frozen_v6():
    """The packaged file is results/v6/bidir_seed1/model.pt, the model scored once on test."""
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(inference.V6_WEIGHTS) == digest(paths.REPO_ROOT / "results" / "v6" / "bidir_seed1" / "model.pt")


def test_classes_major_then_minor():
    assert len(inference.CLASSES) == 24
    assert inference.CLASSES[0] == "C:maj" and inference.CLASSES[9] == "A:maj"
    assert inference.CLASSES[12] == "C:min" and inference.CLASSES[21] == "A:min"


def write_chord(path, freqs, seconds=6.0, sr=22050):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, sum(0.2 * np.sin(2 * np.pi * f * t) for f in freqs), sr)
    return path


@pytest.fixture(scope="module")
def model():
    return inference.load_v6()


def test_posteriors_of_a_file(tmp_path, model):
    """One distribution over the 24 chords per frame, every hop / sr seconds from 0."""
    times, post = inference.posteriors(write_chord(tmp_path / "c.wav", [261.63, 329.63, 392.0]), model=model)
    hop = inference.SETTINGS["cqt"]["hop_length"] / inference.SETTINGS["audio"]["sr"]
    assert post.shape == (24, len(times))
    np.testing.assert_allclose(times, np.arange(len(times)) * hop)
    np.testing.assert_allclose(post.sum(axis=0), 1.0, atol=1e-5)


def test_a_c_major_triad_reads_as_c_major(tmp_path, model):
    _, post = inference.posteriors(write_chord(tmp_path / "c.wav", [130.81, 261.63, 329.63, 392.0]), model=model)
    assert np.mean(post.argmax(axis=0) == inference.CLASSES.index("C:maj")) > 0.8


def test_phases_interleave_on_a_finer_grid(tmp_path, model):
    """Read 4 times, each delayed by a quarter hop: one frame every hop / 4, the first pass's frames unchanged."""
    path = write_chord(tmp_path / "c.wav", [261.63, 329.63, 392.0])
    times, post = inference.posteriors(path, model=model)
    fine_times, fine = inference.posteriors(path, phases=4, model=model)
    step = times[1] / 4
    np.testing.assert_allclose(fine_times, np.arange(len(fine_times)) * step, atol=1e-9)
    np.testing.assert_allclose(fine[:, ::4][:, :len(times)], post, atol=1e-5)


def test_phases_must_divide_the_hop(tmp_path, model):
    with pytest.raises(ValueError):
        inference.posteriors(write_chord(tmp_path / "c.wav", [261.63]), phases=3, model=model)


def flat(freqs, cents):
    return [f * 2 ** (cents / 1200) for f in freqs]


C_MAJOR = [130.81, 261.63, 329.63, 392.0]


def test_a_recording_40_cents_flat_still_reads_as_c_major(tmp_path, model):
    """Tuning is estimated on the semitone, +-50 cents: on V6's quarter-tone bins it would wrap at +-25 cents and
    put the notes between the bins the model reads (Another Girl, 44 cents flat, read a semitone low)."""
    _, post = inference.posteriors(write_chord(tmp_path / "c.wav", flat(C_MAJOR, -40)), model=model)
    assert np.mean(post.argmax(axis=0) == inference.CLASSES.index("C:maj")) > 0.8


def test_the_callers_tuning_is_used(tmp_path, model):
    """tuning in semitones from A440: a C major 40 cents flat reads as C major at its tuning, not when read as in tune."""
    path = write_chord(tmp_path / "f.wav", flat(C_MAJOR, -40))
    c_major = inference.CLASSES.index("C:maj")
    _, right = inference.posteriors(path, tuning=-0.4, model=model)
    _, wrong = inference.posteriors(path, tuning=0.0, model=model)
    assert np.mean(right.argmax(axis=0) == c_major) > 0.8
    assert np.mean(wrong.argmax(axis=0) == c_major) < 0.2
