"""Behaviour of the causal log-CQT: no column reads the future (beyond the one harmonic frame), and each
causal step reduces to V5's at the end of the song."""
import librosa
import numpy as np
import pytest

from chordtracker import features_v7_1 as f

SR, HOP = 22050, 2048
CFG = {"audio": {"sr": SR, "mono": True, "harmonic_margin": 4.0},
       "cqt": {"hop_length": HOP, "fmin": "F#0", "bins_per_octave": 24, "n_bins": 168, "tuning_seconds": 10.0},
       "compression": {"enabled": True, "level_percentile": 99, "gamma": 10}}


def synthetic(seconds=14.0, seed=0, cents=0.0):
    """Tones from C1 to C7 that change every 0.7 s, plus noise bursts: energy in every bin, sudden changes."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    step = int(0.7 * SR)
    for start in range(0, n, step):
        seg = slice(start, min(start + step, n))
        for midi in rng.choice(np.arange(24, 97), size=4, replace=False):
            y[seg] += np.sin(2 * np.pi * librosa.midi_to_hz(midi + cents / 100) * t[seg])
        burst = slice(start, min(start + 400, n))
        y[burst] += rng.normal(0, 2, burst.stop - burst.start)
    return (0.1 * y).astype(np.float32)


def features_of(y):
    return f.causal_log_cqt_of_audio(y, CFG)


@pytest.mark.parametrize("cut_s", [3.0, 6.5, 12.5])
def test_no_column_reads_the_future(cut_s):
    """Columns computed from the audio cut at a time equal the full song's, up to the frame the cut allows.

    Column t needs the audio up to frame t + 1 (the harmonic part's STFT window), so with the audio
    cut at sample T, every column t with (t + 1 + 1) * hop <= T must already be final.
    """
    y = synthetic()
    cut = int(cut_s * SR)
    full, part = features_of(y), features_of(y[:cut])
    final = cut // HOP - 1 - f.LOOKAHEAD_FRAMES
    np.testing.assert_allclose(part[:, :final], full[:, :final], atol=1e-3)


def test_the_cut_would_show_a_centred_pipeline():
    """The check has teeth: V5's centred CQT on the same audio does change before the cut."""
    y = synthetic()
    cut = int(6.5 * SR)
    centred = lambda x: np.abs(librosa.cqt(x, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("F#0"),
                                           n_bins=168, bins_per_octave=24))
    final = cut // HOP - 1 - f.LOOKAHEAD_FRAMES
    assert not np.allclose(centred(y[:cut])[:, :final], centred(y)[:, :final], atol=1e-3)


def test_delayed_bins_are_the_centred_cqt_shifted():
    C = np.arange(12.0).reshape(3, 4)
    out = f.delay_bins(C, np.array([0, 1, 2]))
    np.testing.assert_array_equal(out, [[0, 1, 2, 3], [0, 4, 5, 6], [0, 0, 8, 9]])


def test_low_bins_wait_longer_than_high_bins():
    d = f.bin_delays(CFG)
    assert d[12] == 6 and d.min() == 1 and np.all(np.diff(d) <= 0)    # C1 (row 12): 6 frames


def test_the_running_level_ends_at_the_whole_song_level():
    C = np.random.default_rng(1).random((168, 50))
    level = f.running_level(C, 99)
    assert level[-1] == pytest.approx(np.percentile(C, 99))
    assert level[0] == pytest.approx(np.percentile(C[:, 0], 99))


def test_the_causal_median_only_keeps_a_note_once_it_has_lasted():
    """A tone that starts mid-clip is masked out at first and kept once it fills half the median window."""
    n = int(4 * SR)
    t = np.arange(n) / SR
    y = np.where(t >= 2.0, np.sin(2 * np.pi * 440 * t), 0.0).astype(np.float32)
    h = f.causal_harmonic(y, SR, CFG)
    rms = lambda a, b: float(np.sqrt(np.mean(h[int(a * SR):int(b * SR)] ** 2)))
    assert rms(2.0, 2.1) < 0.2 * rms(3.0, 3.5)
    assert rms(2.6, 2.8) > 0.8 * rms(3.0, 3.5)


@pytest.mark.slow
def test_no_column_reads_the_future_on_real_training_songs():
    """The prefix check on 3 RWC-P and 3 GuitarSet training songs (first 40 s, cut at 15, 25 and 35 s),
    and each song's cached array against a fresh computation."""
    import yaml
    from chordtracker import data_v5, paths
    cfg = yaml.safe_load((paths.REPO_ROOT / "configs" / "v2.yaml").read_text())
    cfg.update(yaml.safe_load((paths.REPO_ROOT / "configs" / "v7_1.yaml").read_text()))
    for take in data_v5.rwc_takes("train")[:3] + data_v5.guitarset_takes("train")[:3]:
        y, _ = librosa.load(str(take.audio), sr=SR, mono=True, duration=40.0)
        full = f.causal_log_cqt_of_audio(y, cfg)
        for cut_s in (15.0, 25.0, 35.0):
            cut = int(cut_s * SR)
            part = f.causal_log_cqt_of_audio(y[:cut], cfg)
            final = cut // HOP - 1 - f.LOOKAHEAD_FRAMES
            np.testing.assert_allclose(part[:, :final], full[:, :final], atol=1e-3, err_msg=take.key)
        cached = f.cached_causal_log_cqt(take.audio, take.key, cfg)
        fresh = f.causal_log_cqt(take.audio, cfg)
        np.testing.assert_allclose(cached, fresh, atol=1e-2, err_msg=take.key)      # cache is float16


def test_the_tuning_is_estimated_on_the_first_seconds_and_used_after():
    """A clip tuned 20 cents sharp: the estimate from its first 10 s finds it, A440 is used before then."""
    y = synthetic(cents=20.0)
    tuning, first = f.start_tuning(y, CFG)
    assert tuning * 50 == pytest.approx(20.0, abs=5.0)                  # 1 bin = 50 cents
    assert first == int(np.ceil(10.0 * SR / HOP)) - 1
    assert f.start_tuning(y[:int(9 * SR)], CFG) is None


def test_a_shorter_median_keeps_a_new_note_sooner():
    """With a 9-frame median a new tone counts as harmonic after ~0.1 s instead of ~0.37 s."""
    n = int(4 * SR)
    t = np.arange(n) / SR
    y = np.where(t >= 2.0, np.sin(2 * np.pi * 440 * t), 0.0).astype(np.float32)
    short = {**CFG, "audio": {**CFG["audio"], "harmonic_median_frames": 9}}
    rms = lambda h, a, b: float(np.sqrt(np.mean(h[int(a * SR):int(b * SR)] ** 2)))
    h31, h9 = f.causal_harmonic(y, SR, CFG), f.causal_harmonic(y, SR, short)
    assert rms(h9, 2.15, 2.3) > 0.8 * rms(h9, 3.0, 3.5)
    assert rms(h31, 2.15, 2.3) < 0.2 * rms(h31, 3.0, 3.5)
    assert f.cache_dir(short).name == "v7_1_logcqt_median9" and f.cache_dir(CFG).name == "v7_1_logcqt"
