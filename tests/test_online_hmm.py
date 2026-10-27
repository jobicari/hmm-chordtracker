"""Behaviour of V7's online decoders: what they may read, what they reduce to, and that live = batch."""
import librosa
import numpy as np
import pytest

from chordtracker import online_hmm as o
from chordtracker import transitions


def random_song(T=60, n=24, seed=0):
    """Posteriors that hold a chord for a while, then change, with noise; a random-ish Q and prior."""
    rng = np.random.default_rng(seed)
    chords = np.repeat(rng.integers(0, n, T // 6 + 1), 6)[:T]
    post = rng.dirichlet(np.ones(n), size=T).T
    post[chords, np.arange(T)] += 1.0
    post /= post.sum(axis=0)
    Q = rng.random((n, n)) + 0.1
    np.fill_diagonal(Q, 0)
    Q /= Q.sum(axis=1, keepdims=True)
    prior = rng.dirichlet(np.ones(n) * 5)
    return post, Q, prior


def stream(decoder, post, prior):
    """Run a streaming decoder over a song: the decision for every frame, in frame order."""
    out = [decoder.step(log_b) for log_b in o.evidence(post, prior).T]
    out = [s for s in out if s is not None]
    return np.array(out + (decoder.flush() if hasattr(decoder, "flush") else []))


@pytest.mark.parametrize("p_self", [0.5, 0.9])
def test_a_lag_as_long_as_the_song_is_offline_viterbi(p_self):
    post, Q, prior = random_song()
    A = transitions.transition(Q, p_self)
    offline = librosa.sequence.viterbi_discriminative(post, A, p_state=prior)
    assert np.array_equal(o.fixed_lag_viterbi(post, A, prior, [post.shape[1] - 1])[post.shape[1] - 1], offline)


def test_streaming_and_batch_decide_the_same():
    post, Q, prior = random_song(seed=1)
    A = transitions.transition(Q, 0.8)
    batch = o.fixed_lag_viterbi(post, A, prior, [0, 1, 3, 8, 100])
    for lag, path in batch.items():
        assert np.array_equal(stream(o.FixedLagViterbi(A, lag), post, prior), path), lag
    assert np.array_equal(stream(o.ForwardFilter(A), post, prior), o.forward_filter(post, A, prior))


@pytest.mark.parametrize("lag", [0, 2, 5])
def test_a_decision_never_reads_past_its_lag(lag):
    """Changing every frame after t + L leaves the decisions for frames 0..t as they were."""
    post, Q, prior = random_song(seed=2)
    A = transitions.transition(Q, 0.9)
    t = 30
    other = post.copy()
    other[:, t + lag + 1:] = np.random.default_rng(9).dirichlet(np.ones(24), size=post.shape[1] - t - lag - 1).T
    for decode in (lambda p: o.fixed_lag_viterbi(p, A, prior, [lag])[lag],
                   lambda p: o.forward_filter(p, A, prior)):
        assert np.array_equal(decode(post)[:t + 1], decode(other)[:t + 1])


def test_a_longer_lag_can_revise_a_decision():
    """The point of waiting: some frame decided differently at lag 0 and at a long lag."""
    post, Q, prior = random_song(seed=3)
    A = transitions.transition(Q, 0.95)
    paths = o.fixed_lag_viterbi(post, A, prior, [0, 20])
    assert (paths[0] != paths[20]).any()


def test_no_transition_preference_gives_the_argmax_of_the_evidence():
    """p_self = 1/24 with a uniform Q makes every row of A uniform: each frame is decided on its own."""
    post, _, prior = random_song(seed=4)
    A = np.full((24, 24), 1 / 24)
    argmax = o.evidence(post, prior).argmax(axis=0)
    assert np.array_equal(o.forward_filter(post, A, prior), argmax)
    for path in o.fixed_lag_viterbi(post, A, prior, [0, 4]).values():
        assert np.array_equal(path, argmax)


def test_a_song_shorter_than_the_lag_is_decided_at_its_end():
    post, Q, prior = random_song(T=5, seed=5)
    A = transitions.transition(Q, 0.7)
    offline = librosa.sequence.viterbi_discriminative(post, A, p_state=prior)
    assert np.array_equal(stream(o.FixedLagViterbi(A, 10), post, prior), offline)
    assert np.array_equal(o.fixed_lag_viterbi(post, A, prior, [10])[10], offline)
