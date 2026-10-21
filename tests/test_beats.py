"""Tests for chordtracker.beats (component 7, V1).

Unit tests use small hand-made arrays and need no audio. The test marked `slow` runs the real
functions on one GuitarSet train take (no val or test take is touched): run it with `pytest -m slow`.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from chordtracker import beats
from chordtracker.data.data import ChordAnnotation

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "v1.yaml").read_text())
MANIFEST = Path(__file__).resolve().parents[1] / "data" / "processed" / "guitarset" / "manifest.csv"
LABELS = ["C:maj", "G:maj", "N"]


def beat_evidence(posteriors, edges, reference_segments, song_id="take"):
    """BeatEvidence with the given edges (seconds) and ((start, end, label), ...) reference."""
    intervals = np.array([[start, end] for start, end, _ in reference_segments], dtype=float)
    reference = ChordAnnotation(intervals, [label for _, _, label in reference_segments])
    return beats.BeatEvidence(song_id, np.asarray(posteriors, dtype=float), np.asarray(edges, dtype=float),
                              reference, float(intervals[-1, 1]))


# ------------------------------------------------------------------------ segments

def test_segment_bounds_cover_every_frame_once():
    """0 and n_frames are added, beats outside (0, n_frames) and duplicates are dropped."""
    assert beats.segment_bounds(np.array([3, 7, 7, 12]), 10).tolist() == [0, 3, 7, 10]
    assert beats.segment_bounds(np.array([0, 5]), 10).tolist() == [0, 5, 10]
    assert beats.segment_bounds(np.array([], dtype=int), 4).tolist() == [0, 4]


def test_beat_sync_pools_each_segment():
    """One column per segment, equal to the mean (or median) of exactly that segment's frames."""
    chroma = np.tile([0.0, 1.0, 2.0, 3.0, 4.0, 10.0], (12, 1))  # every row is the same 6 frames
    bounds = np.array([0, 2, 6])                                 # segments: frames 0-1 and 2-5
    assert np.allclose(beats.beat_sync(chroma, bounds, "mean")[0], [0.5, 4.75])
    assert np.allclose(beats.beat_sync(chroma, bounds, "median")[0], [0.5, 3.5])
    assert beats.beat_sync(chroma, bounds).shape == (12, 2)


def test_beat_sync_rejects_an_unknown_aggregate():
    with pytest.raises(ValueError):
        beats.beat_sync(np.ones((12, 4)), np.array([0, 4]), "max")


def test_segments_to_annotation_merges_runs_at_exact_edges():
    """Boundaries land on the edges themselves, not halfway between them."""
    annotation = beats.segments_to_annotation(np.array([0, 0, 1]), LABELS, np.array([0.0, 0.5, 1.0, 1.5]))
    assert annotation.labels == ["C:maj", "G:maj"]
    assert np.allclose(annotation.intervals, [[0.0, 1.0], [1.0, 1.5]])


def test_segments_to_annotation_needs_one_more_edge_than_segments():
    with pytest.raises(ValueError):
        beats.segments_to_annotation(np.array([0, 1]), LABELS, np.array([0.0, 1.0]))


# ------------------------------------------------------------------------ sweep

def test_sweep_uniform_p_self_matches_framewise_and_viterbi_smooths_a_blip():
    """5 beats of C:maj with a G:maj blip on beat 2: framewise follows it, 1/3 equals framewise, 0.9 holds C:maj."""
    posteriors = np.tile([[0.6], [0.3], [0.1]], (1, 5))
    posteriors[:, 2] = [0.3, 0.6, 0.1]
    take = beat_evidence(posteriors, [0, 1, 2, 3, 4, 5], [(0, 5, "C:maj")])
    summary_rows, per_song_rows = beats.sweep_p_self([take], LABELS, [1 / 3, 0.9])
    framewise, uniform, sticky = summary_rows
    assert [row["system"] for row in summary_rows] == ["framewise", 1 / 3, 0.9]
    assert len(per_song_rows) == 3
    assert framewise["wcsr"] == pytest.approx(0.8) and uniform["wcsr"] == framewise["wcsr"]
    assert sticky["wcsr"] == pytest.approx(1.0)
    assert sticky["mean_changes_per_min"] < framewise["mean_changes_per_min"]


# ------------------------------------------------------------------------ real audio

@pytest.mark.slow
@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest not found in data/processed/guitarset")
def test_real_train_take_runs_through_the_beat_sweep():
    """On train take 00_BN1-129-Eb_comp: beats are found, edges cover the take, 1/25 equals framewise."""
    from chordtracker import emission
    from chordtracker.data.data import load_manifest
    from chordtracker.features import ChromaCQTExtractor

    record = next(r for r in load_manifest(MANIFEST) if r.song_id == "00_BN1-129-Eb_comp")
    model = emission.TemplateEmission.from_config(CFG)
    take = beats.take_evidence(record, ChromaCQTExtractor.from_config(CFG), model, CFG)
    assert take.posteriors.shape == (25, len(take.edges) - 1)
    assert take.edges[0] == 0.0 and take.edges[-1] == take.duration and np.all(np.diff(take.edges) > 0)
    assert len(take.edges) > 20                                   # a 22 s take at 129 BPM has about 47 beats

    summary_rows, _ = beats.sweep_p_self([take], model.labels, [1 / 25, 0.5])
    framewise, uniform, sticky = summary_rows
    assert all(0.0 <= row["wcsr"] <= 1.0 for row in summary_rows)
    assert uniform["wcsr"] == framewise["wcsr"]
    assert sticky["mean_changes_per_min"] < framewise["mean_changes_per_min"]
