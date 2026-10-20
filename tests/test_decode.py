"""Tests for chordtracker.decode (component 4).

Unit tests use hand-made posteriors where the right path is known, plus random chroma for the
p_self = 1/25 check. The tests marked `slow` check behaviour on a real
GuitarSet take (the pipeline runs end to end and Viterbi smooths): run them with `pytest -m slow`.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from chordtracker import decode, emission

GUITARSET = Path(__file__).resolve().parents[1] / "data" / "guitarset"
CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "v0.yaml").read_text())


def blip_posteriors(n_frames=20, blip_frame=10, n_states=3):
    """(3, 20) posteriors: state 0 is likely (0.6) everywhere except one frame where state 1 is (0.6)."""
    posteriors = np.full((n_states, n_frames), 0.4 / (n_states - 1))
    posteriors[0] = 0.6
    posteriors[:, blip_frame] = 0.4 / (n_states - 1)
    posteriors[1, blip_frame] = 0.6
    return posteriors


# ------------------------------------------------------------------------ interface

def test_decoder_without_decode_cannot_be_created():
    """A subclass that forgets decode() raises TypeError at creation, thanks to @abstractmethod."""
    class Incomplete(decode.Decoder):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_from_config():
    """FramewiseDecoder needs no settings; ViterbiDecoder reads p_self = 0.9 from cfg["decoder"]."""
    assert isinstance(decode.FramewiseDecoder.from_config(CFG), decode.FramewiseDecoder)
    assert decode.ViterbiDecoder.from_config(CFG).p_self == CFG["decoder"]["p_self"] == 0.9


@pytest.mark.parametrize("p_self", [0.0, 1.0, 1.5, -0.1])
def test_viterbi_rejects_p_self_outside_0_1(p_self):
    """p_self = 0 forbids staying, 1 forbids changing, 1.5 and -0.1 are not probabilities: all raise ValueError."""
    with pytest.raises(ValueError):
        decode.ViterbiDecoder(p_self)


# ------------------------------------------------------------------------ framewise

def test_framewise_picks_each_column_max():
    """[[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]] -> [0, 1]; a tie [0.5, 0.5, 0] goes to the lowest index, 0."""
    posteriors = np.array([[0.6, 0.2, 0.5],
                           [0.3, 0.5, 0.5],
                           [0.1, 0.3, 0.0]])
    path = decode.FramewiseDecoder().decode(posteriors)
    assert path.shape == (3,)
    assert path.tolist() == [0, 1, 0]


# ------------------------------------------------------------------------ viterbi

def test_transition_matrix():
    """25 states, p_self 0.9: diagonal 0.9, every other entry 0.1 / 24, rows sum to 1."""
    transition = decode.ViterbiDecoder(0.9).transition(25)
    assert transition.shape == (25, 25)
    np.testing.assert_allclose(np.diag(transition), 0.9)
    np.testing.assert_allclose(transition[0, 1:], 0.1 / 24)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)


def test_viterbi_removes_a_one_frame_blip_that_framewise_keeps():
    """State 1 wins frame 10 only: framewise switches 0 -> 1 -> 0, Viterbi at p_self 0.9 stays on 0 throughout."""
    posteriors = blip_posteriors()
    framewise_path = decode.FramewiseDecoder().decode(posteriors)
    viterbi_path = decode.ViterbiDecoder(0.9).decode(posteriors)
    assert framewise_path[10] == 1 and np.count_nonzero(np.diff(framewise_path)) == 2
    assert viterbi_path.tolist() == [0] * 20


def test_viterbi_keeps_a_long_chord_change():
    """State 0 for frames 0..9, state 1 for 10..19 (0.6 vs 0.2 each): Viterbi at p_self 0.9 switches once, at frame 10."""
    posteriors = np.full((3, 20), 0.2)
    posteriors[0, :10] = posteriors[1, 10:] = 0.6
    viterbi_path = decode.ViterbiDecoder(0.9).decode(posteriors)
    assert viterbi_path.tolist() == [0] * 10 + [1] * 10


def test_viterbi_at_uniform_transitions_equals_framewise():
    """p_self = 1/25 makes every transition 0.04, so Viterbi must return the framewise path exactly (500 random frames)."""
    chroma = np.random.default_rng(0).random((12, 500))
    posteriors = emission.TemplateEmission.from_config(CFG).posteriors(chroma)
    framewise_path = decode.FramewiseDecoder().decode(posteriors)
    viterbi_path = decode.ViterbiDecoder(1 / 25).decode(posteriors)
    np.testing.assert_array_equal(viterbi_path, framewise_path)


# ------------------------------------------------------------------------ real data

@pytest.fixture(scope="module")
def real_take():
    """Posteriors (25, 962) and labels of take 00_BN1-129-Eb_comp, computed once for both slow tests."""
    from chordtracker.features import ChromaCQTExtractor

    take = "00_BN1-129-Eb_comp"
    chroma = ChromaCQTExtractor.from_config(CFG).extract(GUITARSET / "audio_mono-mic_comp" / f"{take}_mic.wav")
    model = emission.TemplateEmission.from_config(CFG)
    return model.posteriors(chroma), model.labels


@pytest.mark.slow
@pytest.mark.skipif(not GUITARSET.is_dir(), reason="GuitarSet not found in data/guitarset")
def test_real_take_viterbi_runs_end_to_end_and_smooths(real_take):
    """On take 00_BN1-129-Eb_comp, Viterbi at p_self 0.99 returns one valid state per frame and changes chord less often than framewise.

    No accuracy is pinned here: exact numbers are a property of the take, not of the decoder. At low
    p_self (e.g. 0.5) many paths tie, and a 1e-16 rounding difference in the posteriors picks a
    different one (42.62% in the notebook, 44.07% with TemplateEmission, same 37.63 changes/min).
    """
    posteriors, labels = real_take
    framewise_path = decode.FramewiseDecoder().decode(posteriors)
    viterbi_path = decode.ViterbiDecoder(0.99).decode(posteriors)
    assert viterbi_path.shape == framewise_path.shape == (posteriors.shape[1],)
    assert viterbi_path.min() >= 0 and viterbi_path.max() < len(labels)
    assert np.count_nonzero(np.diff(viterbi_path)) < np.count_nonzero(np.diff(framewise_path))


@pytest.mark.slow
@pytest.mark.skipif(not GUITARSET.is_dir(), reason="GuitarSet not found in data/guitarset")
def test_real_take_viterbi_at_uniform_transitions_equals_framewise(real_take):
    """On the real take too, p_self = 1/25 gives the framewise path frame for frame (962 frames)."""
    posteriors = real_take[0]
    np.testing.assert_array_equal(decode.ViterbiDecoder(1 / 25).decode(posteriors),
                                  decode.FramewiseDecoder().decode(posteriors))
