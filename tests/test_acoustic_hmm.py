"""Behaviour of the V5 glue between CNN frame posteriors and the V4 beat decoder."""
import numpy as np

from chordtracker import acoustic_hmm as ah


def one_hot_posteriors(classes, n=24, confidence=0.9):
    """(n, len(classes)) posteriors that put `confidence` on each frame's class, the rest spread evenly."""
    P = np.full((n, len(classes)), (1 - confidence) / (n - 1))
    P[classes, np.arange(len(classes))] = confidence
    return P


def test_frame_times_are_hop_spaced():
    assert np.allclose(ah.frame_times(3, 22050, 2048), [0.0, 2048 / 22050, 4096 / 22050])


def test_beat_posteriors_are_distributions_one_per_beat():
    times = ah.frame_times(20, 22050, 2048)
    edges = np.array([0.0, 0.5, 1.0, times[-1] + 0.05])
    beats = ah.frames_to_beats(one_hot_posteriors([3] * 20), times, edges)
    assert beats.shape == (24, 3)
    assert np.allclose(beats.sum(axis=0), 1)
    assert (beats.argmax(axis=0) == 3).all()


def test_each_beat_follows_its_own_frames():
    times = ah.frame_times(20, 22050, 2048)
    split = times[10] - 0.01                      # frames 0-9 in beat 0, frames 10-19 in beat 1
    beats = ah.frames_to_beats(one_hot_posteriors([0] * 10 + [7] * 10), times, np.array([0.0, split, 2.0]))
    assert list(beats.argmax(axis=0)) == [0, 7]


def test_two_disagreeing_frames_tie():
    beats = ah.frames_to_beats(one_hot_posteriors([0, 7], confidence=0.8), np.array([0.1, 0.2]),
                               np.array([0.0, 1.0]))
    assert np.isclose(beats[0, 0], beats[7, 0])


def test_a_beat_shorter_than_a_hop_takes_its_nearest_frame():
    times = ah.frame_times(3, 22050, 2048)        # 0.0, 0.093, 0.186
    edges = np.array([0.0, 0.10, 0.12, 0.3])      # beat 1 has no frame centre inside
    beats = ah.frames_to_beats(one_hot_posteriors([0, 5, 9]), times, edges)
    assert beats[:, 1].argmax() == 5              # 0.093 is nearest to the beat's middle, 0.11


def test_training_prior_keeps_only_the_major_minor_balance():
    prior = ah.training_prior(np.array([0, 0, 12, -1]))
    assert np.isclose(prior.sum(), 1)
    assert np.allclose(prior[:12], 2 / 36) and np.allclose(prior[12:], 1 / 36)


def test_decode_with_clear_evidence_follows_it():
    Q = np.full((24, 24), 1 / 23)
    np.fill_diagonal(Q, 0)
    beats = one_hot_posteriors([0, 0, 7, 7], confidence=0.95)
    path = ah.decode(beats, Q, p_self=0.5, prior=np.full(24, 1 / 24))
    assert list(path) == [0, 0, 7, 7]
