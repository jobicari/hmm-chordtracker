"""Tests for chordtracker.features (component 2): load_audio and chroma.

Interface agreed in the thread:
    load_audio(path, cfg) -> (y, sr)    mono, resampled to cfg["audio"]["sr"], harmonic part only
    chroma(y, sr, cfg)    -> (12, T)    chroma_cqt with the settings in cfg["features"]

All tests read the real configs/v0.yaml, so they check the settings V0 actually runs with.
Audio is synthetic (a C major triad made of sines), so the expected answer is known exactly
and the suite needs no dataset. Each test skips if the function it needs does not exist yet.
The test marked `slow` compares against the V0 notebook on a real GuitarSet take: `pytest -m slow`.
"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from chordtracker import features

GUITARSET = Path(__file__).resolve().parents[1] / "data" / "guitarset"
CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "v0.yaml").read_text())
SR = CFG["audio"]["sr"]
HOP = CFG["features"]["hop_length"]


# ------------------------------------------------------------------------ helpers

def need(name):
    """Return features.<name>, or skip the calling test if that function is not written yet.

    Keeps the suite green while features.py is being filled in: a missing function shows up
    as a skip with a reason, not as an AttributeError.
    """
    fn = getattr(features, name, None)
    if fn is None:
        pytest.skip(f"features.{name} not written yet")
    return fn


def c_major(seconds=2.0, sr=SR):
    """C major triad as a float32 waveform: sines at C4, E4 and G4 (pitch classes 0, 4 and 7).

    Divided by 3 so the sum of three unit sines stays within [-1, 1].
    2 s at 22050 Hz gives 44100 samples.
    """
    t = np.arange(int(seconds * sr)) / sr
    return (sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.00)) / 3).astype(np.float32)


# ------------------------------------------------------------------------ load_audio

def test_load_audio_gives_mono_at_target_rate(tmp_path):
    """A stereo 44.1 kHz file comes back mono at cfg["audio"]["sr"] with the same duration.

    Input on disk: 1 s, shape (44100, 2). Expected in memory: shape (22050,), sr == 22050.
    The length may differ by one sample because of resampling rounding.
    """
    load_audio = need("load_audio")
    sf.write(tmp_path / "x.wav", np.stack([c_major(1.0, 44100)] * 2, axis=1), 44100)    # 1 s stereo, 44.1 kHz
    y, sr = load_audio(tmp_path / "x.wav", CFG)
    assert sr == SR and y.ndim == 1
    assert abs(len(y) - SR) <= 1


# ------------------------------------------------------------------------ chroma

def test_chroma_shape_and_frame_count():
    """chroma returns 12 rows (one per pitch class) and one column per hop, plus the frame at t = 0.

    2 s at 22050 Hz = 44100 samples, hop 512 -> 1 + 44100 // 512 = 87 frames, so shape (12, 87).
    This frame count is what later lines frames up with the chord labels in time.
    """
    chroma = need("chroma")
    y = c_major()
    # frames are centred (librosa pads), so there is one frame per hop plus the one at t = 0
    assert chroma(y, SR, CFG).shape == (12, 1 + len(y) // HOP)


def test_chroma_peaks_on_the_chord_notes():
    """For a C major triad, the 3 strongest pitch classes (averaged over time) are C, E and G.

    Checks that the rows are in template order (0 = C, ..., 11 = B) and that tuning = 0.0
    puts each note in the right bin. A row shift here would make every chord come out wrong.
    """
    chroma = need("chroma")
    c = chroma(c_major(), SR, CFG)
    assert set(np.argsort(c.mean(axis=1))[-3:]) == {0, 4, 7}


# ------------------------------------------------------------------------ extractors

def test_extractor_without_extract_cannot_be_created():
    """A subclass that forgets extract() raises TypeError at creation, thanks to @abstractmethod."""
    class Incomplete(features.FeatureExtractor):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_chroma_cqt_extractor_from_config_reads_every_setting():
    """from_config copies cfg["audio"] and cfg["features"] into attributes, and config() gives them back."""
    extractor = features.ChromaCQTExtractor.from_config(CFG)
    assert (extractor.sr, extractor.hop_length) == (SR, HOP)
    assert extractor.config()["audio"] == CFG["audio"]
    assert extractor.config()["features"] == CFG["features"]


def test_chroma_cqt_extractor_rejects_another_kind():
    """A config with features.kind other than chroma_cqt raises ValueError before any audio is read."""
    other = {**CFG, "features": {**CFG["features"], "kind": "stft_chroma"}}
    with pytest.raises(ValueError):
        features.ChromaCQTExtractor.from_config(other)


def test_chroma_cqt_extractor_matches_the_functions(tmp_path):
    """extract(path) gives exactly chroma(*load_audio(path, cfg), cfg): the class only wraps the functions."""
    sf.write(tmp_path / "x.wav", c_major(), SR)
    extractor = features.ChromaCQTExtractor.from_config(CFG)
    np.testing.assert_array_equal(extractor.extract(tmp_path / "x.wav"),
                                  features.chroma(*features.load_audio(tmp_path / "x.wav", CFG), CFG))


def test_frame_times_are_frame_centres():
    """Frame t is centred at t * hop_length / sr: 0, 512 / 22050 = 0.0232 s, 1024 / 22050 = 0.0464 s."""
    extractor = features.ChromaCQTExtractor.from_config(CFG)
    np.testing.assert_allclose(extractor.frame_times(3), [0.0, HOP / SR, 2 * HOP / SR])


# ------------------------------------------------------------------------ real data

@pytest.mark.slow
@pytest.mark.skipif(not GUITARSET.is_dir(), reason="GuitarSet not found in data/guitarset")
def test_real_take_matches_the_v0_notebook():
    """On take 00_BN1-129-Eb_comp, the extractor gives the same chroma as notebooks/V0_BasicPerFrameLevelReference.

    The notebook's own calls are written out below. Same chroma means the notebook's numbers carry over:
    155.88 changes/min and 43.24% framewise, 21.5 changes/min and 44.91% with Viterbi at p_self 0.99.
    22.32 s of audio at hop 512 gives 962 frames.
    """
    import librosa

    path = GUITARSET / "audio_mono-mic_comp" / "00_BN1-129-Eb_comp_mic.wav"
    audio_array, sr = librosa.load(path, sr=22050, mono=True)
    audio_harmo = librosa.effects.harmonic(audio_array, margin=4)
    notebook_chroma = librosa.feature.chroma_cqt(y=audio_harmo, sr=sr, hop_length=512, threshold=0.25)

    extractor_chroma = features.ChromaCQTExtractor.from_config(CFG).extract(path)
    assert extractor_chroma.shape == notebook_chroma.shape == (12, 962)
    np.testing.assert_allclose(extractor_chroma, notebook_chroma)

