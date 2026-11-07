"""Tests for chordtracker.evaluate (component 5).

Unit tests use tiny hand-made annotations whose right answer can be worked out by hand.
The tests marked `slow` check behaviour on a real GuitarSet take (no pinned notebook numbers):
run them with `pytest -m slow`.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from chordtracker import evaluate
from chordtracker.data.data import ChordAnnotation

GUITARSET = Path(__file__).resolve().parents[1] / "data" / "guitarset"
CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "v0.yaml").read_text())
LABELS = ["C:maj", "G:maj", "A:min"]


def annotation(*segments):
    """annotation((0, 4, "C:maj"), (4, 8, "A:min")) -> ChordAnnotation with those two segments."""
    return ChordAnnotation(np.array([[start, end] for start, end, _ in segments], dtype=float),
                           [label for _, _, label in segments])


# ------------------------------------------------------------------------ segments

def test_path_to_annotation_merges_runs_and_splits_halfway():
    """Path [0, 0, 1, 1] at times 0.0..0.3, duration 0.4 -> [0, 0.15) C:maj, [0.15, 0.4) G:maj."""
    estimate = evaluate.path_to_annotation(np.array([0, 0, 1, 1]), LABELS, np.array([0.0, 0.1, 0.2, 0.3]), 0.4)
    np.testing.assert_allclose(estimate.intervals, [[0.0, 0.15], [0.15, 0.4]])
    assert estimate.labels == ["C:maj", "G:maj"]


def test_path_to_annotation_single_chord_and_one_frame_runs():
    """A constant path is one segment [0, duration); [2, 0, 2] gives three segments with a 0.1 s middle one."""
    times = np.array([0.0, 0.1, 0.2])
    constant = evaluate.path_to_annotation(np.array([1, 1, 1]), LABELS, times, 0.3)
    np.testing.assert_allclose(constant.intervals, [[0.0, 0.3]])
    blip = evaluate.path_to_annotation(np.array([2, 0, 2]), LABELS, times, 0.3)
    np.testing.assert_allclose(blip.intervals, [[0.0, 0.05], [0.05, 0.15], [0.15, 0.3]])
    assert blip.labels == ["A:min", "C:maj", "A:min"]


def test_path_to_annotation_rejects_mismatched_lengths():
    """4 path frames for 3 frame times raises ValueError."""
    with pytest.raises(ValueError):
        evaluate.path_to_annotation(np.array([0, 0, 1, 1]), LABELS, np.array([0.0, 0.1, 0.2]), 0.4)


def test_changes_per_minute():
    """3 segments in 30 s -> 2 changes in half a minute -> 4.0; a single segment -> 0.0."""
    three = annotation((0, 10, "C:maj"), (10, 20, "G:maj"), (20, 30, "C:maj"))
    assert evaluate.changes_per_minute(three, 30.0) == 4.0
    assert evaluate.changes_per_minute(annotation((0, 30, "C:maj")), 30.0) == 0.0


# ------------------------------------------------------------------------ scores

def test_score_weights_by_duration():
    """Reference C 0-4, Am 4-8; estimate C 0-5, Am 5-8: wrong only in 4-5 -> 7 of 8 s, WCSR 0.875."""
    reference = annotation((0, 4, "C:maj"), (4, 8, "A:min"))
    estimate = annotation((0, 5, "C:maj"), (5, 8, "A:min"))
    result = evaluate.score(reference, estimate)
    assert (result.total_s, result.scored_s, result.matched_s, result.excluded_s) == (8.0, 8.0, 7.0, 0.0)
    assert result.wcsr == 0.875


def test_in_v0_vocabulary():
    """Plain maj/min triads and N are in the V0 vocabulary; sevenths, inversions, hdim7 and X are not."""
    assert all(evaluate.in_v0_vocabulary(label) for label in ["C:maj", "Eb:min", "C", "N"])
    assert not any(evaluate.in_v0_vocabulary(label) for label in ["C:7", "C:maj7", "A:min7", "C:maj/3", "B:hdim7", "X"])


def test_score_majmin_reduces_sevenths_and_excludes_the_rest():
    """C:maj7 and A:min7 count as C:maj and A:min (reduced), B:hdim7 has no maj/min reading (excluded).

    Reference G:maj 0-2, C:maj7 2-4, A:min7 4-6, B:hdim7 6-8; estimate G:maj, C:maj, A:min, B:min
    -> 6 of 6 scored s right, 4 s of them reduced, 2 s excluded.
    """
    reference = annotation((0, 2, "G:maj"), (2, 4, "C:maj7"), (4, 6, "A:min7"), (6, 8, "B:hdim7"))
    estimate = annotation((0, 2, "G:maj"), (2, 4, "C:maj"), (4, 6, "A:min"), (6, 8, "B:min"))
    result = evaluate.score(reference, estimate)
    assert (result.scored_s, result.matched_s, result.reduced_s, result.excluded_s) == (6.0, 6.0, 4.0, 2.0)
    assert result.wcsr == 1.0


def test_score_fits_the_estimate_to_the_reference_span():
    """An estimate that ends at 6 s on an 8 s reference is padded with N (wrong), one that runs to 10 s is cut at 8 s."""
    reference = annotation((0, 8, "C:maj"))
    short = evaluate.score(reference, annotation((0, 6, "C:maj")))
    long = evaluate.score(reference, annotation((0, 10, "C:maj")))
    assert (short.total_s, short.matched_s) == (8.0, 6.0)
    assert (long.total_s, long.matched_s) == (8.0, 8.0)


def test_score_root_metric_and_unknown_metric():
    """Root ignores quality: C:min against C:maj is right for root, wrong for majmin; "tetrads" raises."""
    reference, estimate = annotation((0, 4, "C:maj")), annotation((0, 4, "C:min"))
    assert evaluate.score(reference, estimate, "root").wcsr == 1.0
    assert evaluate.score(reference, estimate, "majmin").wcsr == 0.0
    with pytest.raises(ValueError):
        evaluate.score(reference, estimate, "tetrads")


def test_wcsr_is_nan_when_nothing_is_scored():
    """A reference made only of B:hdim7 leaves 0 s to score for majmin, so WCSR is nan, not a division error."""
    result = evaluate.score(annotation((0, 4, "B:hdim7")), annotation((0, 4, "B:min")))
    assert result.scored_s == 0.0 and np.isnan(result.wcsr)


def test_pooled_weights_songs_by_duration():
    """15 of 20 s and 5 of 10 s pool to 20 / 30 = 0.667, not the per-song mean 0.625; reduced 4 + 1 = 5 s."""
    songs = [evaluate.SongScore(total_s=20, scored_s=20, matched_s=15, reduced_s=4),
             evaluate.SongScore(total_s=10, scored_s=10, matched_s=5, reduced_s=1)]
    assert evaluate.pooled(songs).wcsr == pytest.approx(2 / 3)
    assert evaluate.pooled(songs).reduced_s == 5


# ------------------------------------------------------------------------ real data

@pytest.fixture(scope="module")
def real_take():
    """Viterbi (p_self 0.99) path, frame times, labels and reference of take 00_BN1-129-Eb_comp."""
    from chordtracker import decode, emission
    from chordtracker.features import ChromaCQTExtractor

    take = "00_BN1-129-Eb_comp"
    extractor = ChromaCQTExtractor.from_config(CFG)
    chroma = extractor.extract(GUITARSET / "audio_mono-mic_comp" / f"{take}_mic.wav")
    model = emission.TemplateEmission.from_config(CFG)
    path = decode.ViterbiDecoder(0.99).decode(model.posteriors(chroma))
    reference = ChordAnnotation.from_jams(GUITARSET / "annotation" / f"{take}.jams", index=0)
    return path, extractor.frame_times(len(path)), model.labels, reference


@pytest.mark.slow
@pytest.mark.skipif(not GUITARSET.is_dir(), reason="GuitarSet not found in data/guitarset")
def test_real_take_reference_against_itself_scores_one(real_take):
    """Every lead-sheet label of the take parses in mir_eval, and the reference scored against itself gives WCSR 1."""
    reference = real_take[3]
    assert evaluate.score(reference, reference).wcsr == 1.0


@pytest.mark.slow
@pytest.mark.skipif(not GUITARSET.is_dir(), reason="GuitarSet not found in data/guitarset")
def test_real_take_estimate_is_consistent(real_take):
    """The Viterbi estimate covers [0, duration] without gaps, has one segment per run of the path, and scores in [0, 1]."""
    path, frame_times, labels, reference = real_take
    duration = reference.intervals[-1, 1]
    estimate = evaluate.path_to_annotation(path, labels, frame_times, duration)
    assert estimate.intervals[0, 0] == 0.0 and estimate.intervals[-1, 1] == duration
    np.testing.assert_array_equal(estimate.intervals[1:, 0], estimate.intervals[:-1, 1])
    assert len(estimate.labels) == np.count_nonzero(np.diff(path)) + 1
    result = evaluate.score(reference, estimate)
    assert 0.0 <= result.matched_s <= result.scored_s <= result.total_s
    assert 0.0 <= result.reduced_s <= result.scored_s
    assert 0.0 <= result.wcsr <= 1.0
