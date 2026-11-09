"""Behaviour of the block-by-block features: the same columns as the whole-file computation, each column as
soon as the audio allows, and close to V7.1's cached (librosa) features."""
import librosa
import numpy as np
import pytest

from chordtracker import features_v7_1, streaming_features as sf
from test_features_v7_1 import CFG, HOP, SR, synthetic

C2 = {**CFG, "audio": {**CFG["audio"], "harmonic_median_frames": 9}, "cqt": {**CFG["cqt"], "fmin": "F#1"}}


def stream(y, cfg, block=HOP):
    s, cols, ready_at = sf.StreamingFeatures(cfg), [], []
    for i in range(0, len(y), block):
        new = s.step(y[i:i + block])
        cols += new
        ready_at += [min(i + block, len(y))] * len(new)
    cols += s.flush()
    return np.stack(cols, axis=1), ready_at


@pytest.mark.parametrize("seconds", [14.0, 6.0])        # past and short of the 10 s tuning point
def test_block_by_block_gives_the_whole_file_columns(seconds):
    y = synthetic(seconds=seconds, cents=15.0)
    streamed, _ = stream(y, C2)
    offline = sf.offline_log_cqt(y, C2)
    assert streamed.shape == offline.shape == (168, 1 + len(y) // HOP)
    np.testing.assert_allclose(streamed, offline, atol=1e-4)


def test_the_block_size_does_not_matter():
    y = synthetic(seconds=12.0)
    np.testing.assert_allclose(stream(y, C2, block=HOP)[0], stream(y, C2, block=777)[0], atol=1e-4)


def test_each_column_comes_out_one_frame_after_its_frame():
    """Column t needs the audio up to frame t + 1 (the harmonic part's STFT window), no more."""
    y = synthetic(seconds=12.0)
    _, ready_at = stream(y, C2)
    for t, sample in enumerate(ready_at[:60]):
        assert sample <= (t + 2) * HOP


def test_the_harmonic_part_is_features_v7_1s():
    y = synthetic(seconds=8.0)
    s = sf.StreamingFeatures(C2)
    for i in range(0, len(y), HOP):
        s.step(y[i:i + HOP])
    s.flush()
    np.testing.assert_allclose(s.y_h.view, features_v7_1.causal_harmonic(y, SR, C2), atol=1e-5)


@pytest.mark.parametrize("block", [HOP, 777, 2 * HOP])
def test_the_tuning_medians_computed_ahead_are_exactly_start_tunings(block):
    """Frame by frame while the first 10 s arrive: bitwise the medians of the whole call, and the same tuning."""
    y = synthetic(seconds=12.0, cents=15.0).astype(np.float32)
    n = int(C2["cqt"]["tuning_seconds"] * SR)
    medians = sf.TuningMedians(C2, n)
    for i in range(0, len(y), block):
        medians.update(y[:i + block])
    mag = np.abs(librosa.stft(y[:n], n_fft=features_v7_1.N_FFT, hop_length=features_v7_1.STFT_HOP))
    ahead = medians.mag.shape[1]
    assert mag.shape[1] - 3 <= ahead < mag.shape[1]          # all but the frames that read zeros after n
    H, P = features_v7_1.harmonic_medians(mag, C2)
    H_ahead, P_ahead = medians(mag, C2)
    assert np.array_equal(H, H_ahead) and np.array_equal(P, P_ahead)
    fresh = sf.TuningMedians(C2, n)
    for i in range(0, len(y), block):
        fresh.update(y[:i + block])
    assert features_v7_1.start_tuning(y, C2, fresh) == features_v7_1.start_tuning(y, C2)


def test_tuning_medians_recompute_frames_that_differ():
    """A frame seen ahead that is not the whole call's (here: changed on purpose) is computed again."""
    y = synthetic(seconds=11.0).astype(np.float32)
    n = int(C2["cqt"]["tuning_seconds"] * SR)
    medians = sf.TuningMedians(C2, n)
    medians.update(y)
    medians.mag[:, 200] += 1.0
    mag = np.abs(librosa.stft(y[:n], n_fft=features_v7_1.N_FFT, hop_length=features_v7_1.STFT_HOP))
    H, P = features_v7_1.harmonic_medians(mag, C2)
    H_ahead, P_ahead = medians(mag, C2)
    assert np.array_equal(H, H_ahead) and np.array_equal(P, P_ahead)


def test_close_to_the_cached_librosa_features_away_from_the_edges():
    """librosa's CQT downsamples the low octaves; the direct CQT differs slightly there."""
    y = synthetic(seconds=14.0)
    direct, cached = sf.offline_log_cqt(y, C2), features_v7_1.causal_log_cqt_of_audio(y, C2)
    inner = slice(10, direct.shape[1] - 5)                   # the first frames' level and the end padding differ
    assert np.abs(direct[:, inner] - cached[:, inner]).mean() < 0.01


def test_the_sorted_percentile_is_numpys():
    x = np.sort(np.random.default_rng(0).random(1001))
    for q in (0, 50, 99, 100):
        assert sf.sorted_percentile(x, q) == pytest.approx(np.percentile(x, q))


@pytest.mark.slow
def test_block_by_block_on_real_training_songs():
    """3 RWC-P and 3 GuitarSet training songs (first 40 s): the streamed columns equal the whole-file direct
    computation, and stay close to the V7.1 cache away from the first frames."""
    import yaml
    from chordtracker import data_v5, paths
    cfg = yaml.safe_load((paths.REPO_ROOT / "configs" / "v2.yaml").read_text())
    cfg.update(yaml.safe_load((paths.REPO_ROOT / "configs" / "v7_2.yaml").read_text()))
    for fmin in ("F#0", "F#1"):
        cfg["cqt"]["fmin"] = fmin
        for take in data_v5.rwc_takes("train")[:3] + data_v5.guitarset_takes("train")[:3]:
            y, _ = librosa.load(str(take.audio), sr=SR, mono=True, duration=40.0)
            streamed, _ = stream(y, cfg)
            np.testing.assert_allclose(streamed, sf.offline_log_cqt(y, cfg), atol=1e-4, err_msg=take.key)
            cached = features_v7_1.causal_log_cqt_of_audio(y, cfg)
            assert np.abs(streamed[:, 10:-5] - cached[:, 10:-5]).mean() < 0.01, take.key


def test_the_live_path_reads_a_tone_40_cents_sharp_on_its_semitone_bin():
    """A4 played 40 cents sharp, block by block: A440 puts its peak a bin (50 cents) above A4 until the tuning
    point; from then on the 12-bin estimate (+40, where 24 bins read -10) puts it on A4's bin."""
    t = np.arange(int(14 * SR)) / SR
    y = (0.1 * np.sin(2 * np.pi * librosa.midi_to_hz(69 + 0.40) * t)).astype(np.float32)
    streamed, _ = stream(y, C2, block=1024)
    a4 = 2 * (69 - librosa.note_to_midi("F#1"))
    first = int(np.ceil(10 * SR / HOP)) - 1
    peaks = streamed.argmax(axis=0)
    assert (peaks[20:first - 2] == a4 + 1).all()
    assert (peaks[first + 2:-4] == a4).all()
